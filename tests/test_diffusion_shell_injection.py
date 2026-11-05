"""Security contract for diffusion quant container plans.

``build_diffusion_docker_run_plan`` must not interpolate ``DiffusionPolicy.name``,
``extras["model_slug"]``, or ``extras["overlay_path"]`` into a shell string.
Those fields are caller-supplied and otherwise unconstrained.

Contract these tests enforce:

- A value is a single safe path segment only when it matches
  ``^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$``. Anything else raises ``ValueError``
  before a plan is returned and before any container command runs.
- ``diffusion_quantize_cli_args`` rejects an unsafe checkpoint name or model
  slug. ``build_diffusion_docker_run_plan`` rejects an unsafe checkpoint name,
  model slug, or overlay path with ``apply_overlay`` either left on or set
  false. Overlay path is unused when ``apply_overlay`` is false.
- When every field is a safe segment, each of the overlay path, model slug,
  and checkpoint path is its own argv element after the image. The plan does
  not invoke a shell (``bash``/``sh``/``dash``/``ash``/``zsh`` or ``-c``) and
  does not join those fields with ``&&``.

The execution helpers below are the caller from the attack: they take the
returned argv, bind-mount every ``-v`` source onto that argv's destination,
and exec the container command that follows ``-w``. On the unfixed planner
that command is ``bash -c`` of the joined string, so a metacharacter in a
policy field runs and the writable mounts carry the write back to the host
directories passed into the planner.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from model_forge.modelopt.diffusion_execution import (
    CONTAINER_EXPORT,
    CONTAINER_HF_CACHE,
    DiffusionDockerRunPlan,
    build_diffusion_docker_run_plan,
    diffusion_quantize_cli_args,
)
from model_forge.modelopt.diffusion_policy import DiffusionPolicy

_IMAGE = "modelopt-quant-test:local"
_SAFE_NAME = "safe-ckpt"
_SAFE_SLUG = "qwen-image-2.1"
_SAFE_OVERLAY = "ok-overlay.py"
_SHELLS = frozenset({"bash", "sh", "dash", "ash", "zsh"})

# Classes of values that are not one safe path segment. The shell cases are the
# ones the joined ``bash -c`` string actually executes; the rest are rejected
# so a later metacharacter cannot be smuggled through the same fields.
_UNSAFE_SEGMENTS = (
    "ok;id",
    "ok|id",
    "ok&id",
    "ok$(id)",
    "`id`",
    "ok\nid",
    "ok id",
    "ok\tid",
    "ok>id",
    "ok<id",
    "ok'id",
    'ok"id',
    "ok\\id",
    "ok*id",
    "ok?id",
    "ok#id",
    "ok/child",
    "..",
    ".",
    "",
    "-ok",
    "configs/modelopt/diffusion/qwen-image-21-overlay.py",
)

_UNSHARE = "/usr/bin/unshare"


def _user_namespaces_available() -> bool:
    """Whether this process may create the user and mount namespaces `_execute_plan` runs plans in."""
    try:
        probe = subprocess.run(
            [_UNSHARE, "--user", "--map-root-user", "--mount", "true"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


requires_user_namespaces = pytest.mark.skipif(
    not _user_namespaces_available(),
    reason=f"executing a plan needs {_UNSHARE} --user --map-root-user --mount, which is missing or "
    "not permitted here (unprivileged container or no user namespaces)",
)

_RUNNER = r"""
set -eu
/usr/bin/mount -t tmpfs tmpfs /mnt
while [ "$1" != "--" ]; do
  dest=$1
  src=$2
  shift 2
  /usr/bin/mkdir -p "$dest"
  /usr/bin/mount --bind "$src" "$dest"
done
shift
shim=$(/usr/bin/mktemp -d)
cat > "$shim/python" <<'PY'
#!/bin/sh
exec /usr/bin/python3 "$@"
PY
/bin/chmod +x "$shim/python"
export PATH="$shim:/usr/bin:/bin"
exec "$@"
"""


def _policy(
    *,
    name: str = _SAFE_NAME,
    slug: str | None = _SAFE_SLUG,
    overlay: str | None = _SAFE_OVERLAY,
    quant_format: str = "fp8",
) -> DiffusionPolicy:
    extras: dict[str, str] = {}
    if slug is not None:
        extras["model_slug"] = slug
    if overlay is not None:
        extras["overlay_path"] = overlay
    return DiffusionPolicy(name=name, quant_format=quant_format, extras=extras)


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    export_dir = tmp_path / "export"
    hf_cache = tmp_path / "cache"
    export_dir.mkdir()
    hf_cache.mkdir()
    return export_dir, hf_cache


def _build(
    export_dir: Path,
    hf_cache: Path,
    policy: DiffusionPolicy,
    *,
    apply_overlay: bool = True,
) -> DiffusionDockerRunPlan:
    return build_diffusion_docker_run_plan(
        docker_bin="docker",
        image=_IMAGE,
        export_dir=export_dir,
        hf_cache=hf_cache,
        policy=policy,
        calib_size=1,
        apply_overlay=apply_overlay,
    )


def _mount_pairs(argv: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    index = 0
    while index < len(argv):
        if argv[index] == "-v" and index + 1 < len(argv):
            source, dest = argv[index + 1].split(":", 1)
            pairs.append((source, dest))
            index += 2
        else:
            index += 1
    return pairs


def _container_command(argv: list[str]) -> list[str]:
    """Return the argv the image runs: the tokens after ``-w WORKDIR IMAGE``."""
    workdir_flag = len(argv) - 1 - argv[::-1].index("-w")
    return argv[workdir_flag + 3 :]


def _execute_plan(plan: DiffusionDockerRunPlan) -> subprocess.CompletedProcess[str]:
    """Run the plan's container command with the plan's bind mounts applied.

    ``docker`` is not installed here. ``docker run -v SRC:DEST IMAGE CMD...``
    bind-mounts SRC onto DEST and execs CMD. This helper does those two steps
    with ``mount --bind`` and ``exec`` of the command taken from the plan, so
    a ``bash -c`` payload in that command is the payload the planner built.
    """
    argv = plan.argv
    pairs = _mount_pairs(argv)
    command = _container_command(argv)
    if not pairs or not command:
        raise AssertionError(f"plan is missing mounts or a container command: {argv!r}")
    runner_args = [_UNSHARE, "--user", "--map-root-user", "--mount"]
    runner_args.extend(["/usr/bin/bash", "-c", _RUNNER, "bash"])
    for source, dest in pairs:
        runner_args.extend([dest, source])
    runner_args.append("--")
    runner_args.extend(command)
    return subprocess.run(
        runner_args,
        check=False,
        env={"PATH": os.pathsep.join(("/usr/bin", "/bin"))},
        capture_output=True,
        text=True,
    )


def _require_rejected(
    plan_factory: Callable[[], DiffusionDockerRunPlan], markers: list[Path]
) -> None:
    try:
        plan = plan_factory()
    except ValueError:
        written = [marker for marker in markers if marker.exists()]
        assert not written, f"policy was rejected only after writing {written}"
        return
    completed = _execute_plan(plan)
    written = [marker for marker in markers if marker.exists()]
    command = _container_command(plan.argv)
    assert not written, (
        "container command executed policy fields through the shell and wrote "
        f"{[str(path) for path in written]}; command={command!r}"
    )
    detail = (completed.stderr or completed.stdout or "")[-500:]
    raise AssertionError(
        "unsafe DiffusionPolicy fields must be rejected with ValueError before a "
        f"plan is returned; container exit {completed.returncode}: {detail}"
    )


def _unsafe_policy(field: str, value: str) -> DiffusionPolicy:
    if field == "name":
        return _policy(name=value)
    if field == "model_slug":
        return _policy(slug=value)
    if field == "overlay_path":
        return _policy(overlay=value)
    raise AssertionError(f"unknown policy field {field}")


@pytest.mark.parametrize("field", ("name", "model_slug", "overlay_path"))
@pytest.mark.parametrize("value", _UNSAFE_SEGMENTS)
def test_unsafe_segment_is_rejected_before_any_write(
    tmp_path: Path, field: str, value: str
) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    policy = _unsafe_policy(field, value)
    if field != "overlay_path":
        with pytest.raises(ValueError):
            diffusion_quantize_cli_args(policy=policy, export_dir=CONTAINER_EXPORT, calib_size=1)
    with pytest.raises(ValueError):
        _build(export_dir, hf_cache, policy)
    assert list(export_dir.iterdir()) == []
    assert list(hf_cache.iterdir()) == []


def test_apply_overlay_false_rejects_unsafe_name_and_slug(tmp_path: Path) -> None:
    """The non-shell branch still receives the same unsanitized name and slug."""
    export_dir, hf_cache = _dirs(tmp_path)
    for policy in (
        _policy(name="ckpt;id"),
        _policy(slug="slug;id"),
    ):
        with pytest.raises(ValueError):
            diffusion_quantize_cli_args(policy=policy, export_dir=CONTAINER_EXPORT, calib_size=1)
        with pytest.raises(ValueError):
            _build(export_dir, hf_cache, policy, apply_overlay=False)
    assert list(export_dir.iterdir()) == []
    assert list(hf_cache.iterdir()) == []


@requires_user_namespaces
@pytest.mark.parametrize("quant_format", ("fp8", "nvfp4"))
def test_checkpoint_name_is_not_executed(tmp_path: Path, quant_format: str) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    export_marker = export_dir / "name-wrote"
    cache_marker = hf_cache / "name-wrote"
    policy = _policy(
        name="ckpt; touch /mnt/export/name-wrote /mnt/hf_cache/name-wrote; ",
        quant_format=quant_format,
    )

    def build() -> DiffusionDockerRunPlan:
        return _build(export_dir, hf_cache, policy)

    _require_rejected(build, [export_marker, cache_marker])


@requires_user_namespaces
def test_model_slug_is_not_executed(tmp_path: Path) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    export_marker = export_dir / "slug-wrote"
    cache_marker = hf_cache / "slug-wrote"
    policy = _policy(slug="slug; touch /mnt/export/slug-wrote /mnt/hf_cache/slug-wrote; ")

    def build() -> DiffusionDockerRunPlan:
        return _build(export_dir, hf_cache, policy)

    _require_rejected(build, [export_marker, cache_marker])


@requires_user_namespaces
def test_overlay_path_metacharacter_is_not_executed(tmp_path: Path) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    export_marker = export_dir / "overlay-wrote"
    cache_marker = hf_cache / "overlay-wrote"
    policy = _policy(overlay="ok.py; touch /mnt/export/overlay-wrote /mnt/hf_cache/overlay-wrote")

    def build() -> DiffusionDockerRunPlan:
        return _build(export_dir, hf_cache, policy)

    _require_rejected(build, [export_marker, cache_marker])


@requires_user_namespaces
def test_overlay_command_substitution_is_not_executed(tmp_path: Path) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    export_marker = export_dir / "dollar-wrote"
    cache_marker = hf_cache / "dollar-wrote"
    policy = _policy(overlay="$(touch /mnt/export/dollar-wrote /mnt/hf_cache/dollar-wrote)")

    def build() -> DiffusionDockerRunPlan:
        return _build(export_dir, hf_cache, policy)

    _require_rejected(build, [export_marker, cache_marker])


@requires_user_namespaces
def test_overlay_newline_is_not_executed(tmp_path: Path) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    export_marker = export_dir / "newline-wrote"
    policy = _policy(overlay="ok.py\ntouch /mnt/export/newline-wrote")

    def build() -> DiffusionDockerRunPlan:
        return _build(export_dir, hf_cache, policy)

    _require_rejected(build, [export_marker])


def test_safe_segments_are_argv_elements_not_a_shell_string(tmp_path: Path) -> None:
    export_dir, hf_cache = _dirs(tmp_path)
    policy = _policy()
    plan = _build(export_dir, hf_cache, policy)
    argv = plan.argv
    tail = argv[argv.index(_IMAGE) + 1 :]
    assert _SAFE_OVERLAY in tail
    assert _SAFE_SLUG in tail
    assert f"{CONTAINER_EXPORT}/{_SAFE_NAME}.pt" in tail
    assert not _SHELLS.intersection(argv)
    assert "-c" not in argv
    assert all("&&" not in part and "$(" not in part and "\n" not in part for part in argv)
    assert f"{export_dir}:{CONTAINER_EXPORT}" in argv
    assert f"{hf_cache}:{CONTAINER_HF_CACHE}" in argv
    assert f"{export_dir}:{CONTAINER_EXPORT}:ro" not in argv
    assert f"{hf_cache}:{CONTAINER_HF_CACHE}:ro" not in argv
    assert list(export_dir.iterdir()) == []
    assert list(hf_cache.iterdir()) == []
