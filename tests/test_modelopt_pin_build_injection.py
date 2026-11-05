"""Pin strings must not run as shell or Python during a ModelOpt image build.

``containers/modelopt/build.sh`` loads ``version``, ``git_commit``, and
``wheel.{url,sha256,filename}`` from a pin file and passes them to
``docker build --build-arg``. The Dockerfiles use ``# syntax=docker/dockerfile:1.7``.
That frontend stores each build-arg as an environment variable and runs a
shell-form ``RUN`` as ``/bin/sh -c`` with the instruction text still containing
``${...}``. These tests drive that path:

- ``build.sh`` must refuse a pin field that is not the recorded shape, and it
  must not start docker while doing so. A second read of the pin file must not
  supply the value that is built.
- Each ``RUN`` that mentions a pin build-arg is executed with ``/bin/sh -c``
  and that value in the environment. The value must not create a file.

The assertions fail while a pin string is still interpolated into a ``python -c``
program or is still forwarded to docker unchanged.
"""

from __future__ import annotations

import copy
import ctypes
import json
import os
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent
BUILD = REPO / "containers" / "modelopt" / "build.sh"
PIN = REPO / "configs" / "modelopt" / "pin.json"
DOCKERFILES = (
    REPO / "containers" / "modelopt" / "Dockerfile",
    REPO / "containers" / "modelopt" / "issue14" / "Dockerfile",
    REPO / "containers" / "modelopt" / "nemotron-3.5-rerank-8b" / "Dockerfile",
)
# linux/sys/inotify.h IN_CLOSE_NOWRITE: a read-only fd was closed.
_IN_CLOSE_NOWRITE = 0x00000010

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.inotify_init1.argtypes = [ctypes.c_int]
_LIBC.inotify_init1.restype = ctypes.c_int
_LIBC.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
_LIBC.inotify_add_watch.restype = ctypes.c_int

_ENV_FOR = {
    "version": "MODELOPT_VERSION",
    "git_commit": "MODELOPT_COMMIT",
    "wheel.url": "MODELOPT_WHEEL_URL",
    "wheel.sha256": "MODELOPT_WHEEL_SHA256",
    "wheel.filename": "MODELOPT_WHEEL_FILENAME",
}
# {version} and {marker} are filled in by the test. The version and filename
# strings close the single-quoted Python text in the Dockerfiles. The commit,
# URL, and digest strings close a double-quoted shell word or expand a command.
_PAYLOADS = (
    ("version", "{version}'; __import__('os').system('touch {marker}'); #"),
    ("git_commit", "\"; touch {marker}; echo \""),
    ("wheel.url", "https://example.invalid/x\"; touch {marker}; echo \""),
    ("wheel.sha256", "$(touch {marker})"),
    (
        "wheel.filename",
        "a-b-c-d-e') or __import__('os').system('touch {marker}') or ('f.whl",
    ),
)


def _dockerfile_id(path: Path) -> str:
    return str(path.relative_to(REPO / "containers"))


def _load_pin() -> dict[str, Any]:
    return dict(json.loads(PIN.read_text(encoding="utf-8")))


def _assign(pin: dict[str, Any], field: str, value: str) -> None:
    if field.startswith("wheel."):
        wheel = dict(pin["wheel"])
        wheel[field.split(".", 1)[1]] = value
        pin["wheel"] = wheel
        return
    pin[field] = value


def _expected_build_args(pin: dict[str, Any]) -> dict[str, str]:
    wheel = dict(pin["wheel"])
    return {
        "MODELOPT_VERSION": str(pin["version"]),
        "MODELOPT_COMMIT": str(pin["git_commit"]),
        "MODELOPT_WHEEL_URL": str(wheel["url"]),
        "MODELOPT_WHEEL_SHA256": str(wheel["sha256"]),
        "MODELOPT_WHEEL_FILENAME": str(wheel["filename"]),
    }


def _logical_lines(text: str) -> list[str]:
    lines: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        buffer += stripped
        lines.append(buffer)
        buffer = ""
    if buffer:
        lines.append(buffer)
    return lines


def _run_scripts(text: str) -> list[str]:
    scripts: list[str] = []
    for line in _logical_lines(text):
        instruction, _, body = line.partition(" ")
        if instruction.upper() == "RUN":
            scripts.append(body)
    return scripts


def _arg_default(text: str, name: str) -> str:
    prefix = f"ARG {name}="
    for line in _logical_lines(text):
        if line.startswith(prefix):
            return line[len(prefix) :]
    raise AssertionError(f"{name} has no default")


def _split_commands(script: str) -> list[str]:
    """Split a shell script on ``;`` that are outside quotes."""
    parts: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    for char in script:
        if escaped:
            buf.append(char)
            escaped = False
            continue
        if char == "\\" and quote != "'":
            buf.append(char)
            escaped = True
            continue
        if quote is not None:
            if char == quote:
                quote = None
            buf.append(char)
            continue
        if char in {"'", '"'}:
            quote = char
            buf.append(char)
            continue
        if char == ";":
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(char)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _install_build_tools(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    bindir = root / "bin"
    bindir.mkdir()
    os.symlink(sys.executable, bindir / "python")
    _write_executable(bindir / "curl", "#!/bin/sh\nexit 0\n")
    _write_executable(bindir / "sha256sum", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bindir / "git",
        "#!/bin/sh\n"
        "if [ \"$1\" = clone ]; then\n"
        "  dest=\n"
        "  for arg in \"$@\"; do dest=$arg; done\n"
        "  mkdir -p \"$dest/examples/hf_ptq\"\n"
        "  : > \"$dest/examples/hf_ptq/hf_ptq.py\"\n"
        "fi\n"
        "if [ \"$1\" = rev-parse ]; then\n"
        "  printf '%s\\n' \"$FAKE_HEAD\"\n"
        "fi\n"
        "exit 0\n",
    )
    package = root / "py"
    (package / "pip").mkdir(parents=True)
    (package / "pip" / "__init__.py").write_text("", encoding="utf-8")
    (package / "pip" / "__main__.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    version = str(_load_pin()["version"])
    filename = str(_load_pin()["wheel"]["filename"])
    (package / "modelopt").mkdir()
    (package / "modelopt" / "__init__.py").write_text(
        f"__version__ = {version!r}\n",
        encoding="utf-8",
    )
    dist = package / "nvidia_modelopt-0.46.0rc2.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: nvidia-modelopt\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist / "direct_url.json").write_text(
        json.dumps({"url": f"file:///opt/wheel/{filename}"}),
        encoding="utf-8",
    )
    return package


def _runtime_env(root: Path, dockerfile_text: str) -> dict[str, str]:
    package = _install_build_tools(root)
    pin = _load_pin()
    torch = _arg_default(dockerfile_text, "TORCH_VERSION")
    torch_dist = package / f"torch-{torch}.dist-info"
    torch_dist.mkdir()
    (torch_dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: torch\nVersion: {torch}\n",
        encoding="utf-8",
    )
    for name in ("transformers", "datasets", "soundfile", "librosa", "pyarrow"):
        module = package / name
        module.mkdir()
        (module / "__init__.py").write_text("__version__ = '5.18.0'\n", encoding="utf-8")
    workspace = root / "ws"
    workspace.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{root / 'bin'}{os.pathsep}{env.get('PATH', '')}",
            "PYTHONPATH": str(package),
            "PUBLIC_WORKSPACE": str(workspace),
            "PUBLIC_ARTIFACT_PATH": str(workspace / "wheel" / "pkg.whl"),
            "VIRTUAL_ENV": str(workspace / "venv"),
            "MODELOPT_VERSION": str(pin["version"]),
            "MODELOPT_COMMIT": str(pin["git_commit"]),
            "MODELOPT_WHEEL_URL": str(pin["wheel"]["url"]),
            "MODELOPT_WHEEL_SHA256": str(pin["wheel"]["sha256"]),
            "MODELOPT_WHEEL_FILENAME": str(pin["wheel"]["filename"]),
            "MODELOPT_REPO_URL": "https://example.invalid/Model-Optimizer.git",
            "MODELOPT_ROOT": str(workspace / "modelopt"),
            "TRANSFORMERS_SPEC": "transformers==5.18.0",
            "TORCH_VERSION": torch,
            "TORCH_INDEX_URL": "https://example.invalid/whl",
            "EXTRA_PIP_SPECS": "example",
            "PYTHON_VERSION": "3.12",
            "FAKE_HEAD": str(pin["git_commit"]),
        }
    )
    return env


def _execute_runs(dockerfile: Path, env: dict[str, str], env_name: str) -> None:
    text = dockerfile.read_text(encoding="utf-8")
    for script in _run_scripts(text):
        if env_name not in script:
            continue
        subprocess.run(["/bin/sh", "-c", script], env=env, check=False)
        for command in _split_commands(script):
            if env_name not in command:
                continue
            subprocess.run(["/bin/sh", "-c", command], env=env, check=False)


def _parse_build_args(blob: bytes) -> dict[str, str]:
    parts = [part.decode() for part in blob.split(b"\0") if part]
    values: dict[str, str] = {}
    index = 0
    while index < len(parts):
        if parts[index] == "--build-arg" and index + 1 < len(parts):
            key, separator, value = parts[index + 1].partition("=")
            if separator != "=":
                raise AssertionError(parts[index + 1])
            values[key] = value
            index += 2
            continue
        index += 1
    return values


def _run_build(
    root: Path,
    pin_text: str,
    *,
    swap_text: str | None = None,
) -> tuple[int, bool, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    bindir = root / "bin"
    bindir.mkdir()
    os.symlink(sys.executable, bindir / "python")
    invoked = root / "docker-invoked"
    log = root / "docker-log"
    _write_executable(
        bindir / "docker",
        "#!/bin/sh\n"
        "touch \"$DOCKER_INVOKED\"\n"
        "printf '%s\\0' \"$@\" > \"$DOCKER_LOG\"\n"
        "exit 0\n",
    )
    pin = root / "pin.json"
    pin.write_text(pin_text, encoding="utf-8")
    stop: threading.Event | None = None
    if swap_text is not None:
        stop = _swap_after_first_read(pin, swap_text)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bindir}{os.pathsep}{env.get('PATH', '')}",
            "PIN_JSON": str(pin),
            "DOCKER": str(bindir / "docker"),
            "DOCKER_INVOKED": str(invoked),
            "DOCKER_LOG": str(log),
            "IMAGE": "local/modelopt:pin-test",
        }
    )
    try:
        completed = subprocess.run(
            ["bash", str(BUILD)],
            env=env,
            cwd=REPO,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    finally:
        if stop is not None:
            stop.set()
    values = _parse_build_args(log.read_bytes()) if log.is_file() else {}
    return completed.returncode, invoked.is_file(), values


def _swap_after_first_read(pin: Path, replacement: str) -> threading.Event:
    stop = threading.Event()
    fd = _LIBC.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "inotify_init1")
    watch = _LIBC.inotify_add_watch(fd, os.fsencode(pin), _IN_CLOSE_NOWRITE)
    if watch < 0:
        os.close(fd)
        raise OSError(ctypes.get_errno(), "inotify_add_watch")

    def watch_reads() -> None:
        swapped = False
        try:
            while not stop.is_set():
                readable, _, _ = select.select([fd], [], [], 0.05)
                if not readable:
                    continue
                os.read(fd, 4096)
                if swapped:
                    continue
                temporary = pin.with_suffix(".next")
                temporary.write_text(replacement, encoding="utf-8")
                os.replace(temporary, pin)
                swapped = True
        finally:
            os.close(fd)

    threading.Thread(target=watch_reads, daemon=True).start()
    return stop


def _alternate_pin(pin: dict[str, Any]) -> dict[str, Any]:
    """A second pin whose every field still has the shape a checker would allow."""
    other = copy.deepcopy(pin)
    other["version"] = "1.2.3"
    other["git_commit"] = "b" * 40
    other["wheel"] = {
        "url": "https://example.invalid/nvidia_modelopt-1.2.3-py3-none-any.whl",
        "sha256": "cd" * 32,
        "filename": "nvidia_modelopt-1.2.3-py3-none-any.whl",
    }
    return other


@pytest.mark.parametrize(("field", "template"), _PAYLOADS)
def test_build_script_refuses_pin_field_before_docker(
    tmp_path: Path,
    field: str,
    template: str,
) -> None:
    marker = tmp_path / "pin-executed"
    pin = _load_pin()
    value = template.format(version=pin["version"], marker=marker)
    _assign(pin, field, value)
    code, invoked, _values = _run_build(tmp_path / "build", json.dumps(pin))
    assert not marker.is_file()
    assert not invoked
    assert code != 0


def test_build_script_forwards_one_snapshot_of_a_recorded_pin(tmp_path: Path) -> None:
    pin = _load_pin()
    code, invoked, values = _run_build(tmp_path / "build", json.dumps(pin))
    assert code == 0
    assert invoked
    assert values == _expected_build_args(pin)


def test_build_script_does_not_reread_the_pin_between_fields(tmp_path: Path) -> None:
    pin = _load_pin()
    other = _alternate_pin(pin)
    code, invoked, values = _run_build(
        tmp_path / "build",
        json.dumps(pin),
        swap_text=json.dumps(other),
    )
    if invoked:
        assert values == _expected_build_args(pin)
        return
    assert code != 0


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=_dockerfile_id)
@pytest.mark.parametrize(("field", "template"), _PAYLOADS)
def test_dockerfile_run_does_not_execute_pin_field(
    tmp_path: Path,
    dockerfile: Path,
    field: str,
    template: str,
) -> None:
    text = dockerfile.read_text(encoding="utf-8")
    env_name = _ENV_FOR[field]
    if not any(env_name in script for script in _run_scripts(text)):
        pytest.skip(f"{env_name} is not expanded by a RUN in {_dockerfile_id(dockerfile)}")
    marker = tmp_path / "pin-executed"
    pin = _load_pin()
    env = _runtime_env(tmp_path / "run", text)
    env[env_name] = template.format(version=pin["version"], marker=marker)
    env["FAKE_HEAD"] = env["MODELOPT_COMMIT"]
    _execute_runs(dockerfile, env, env_name)
    assert not marker.is_file(), f"{_dockerfile_id(dockerfile)} ran the {field} pin string"
