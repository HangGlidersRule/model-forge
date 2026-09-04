"""Runtime snapshot -> restore artifact rendering.

The remote host may be running a serving container that was *not* started from a
compose file we own (the old runner assumed a nonexistent compose path). Instead
of depending on that, we snapshot the live container with ``docker inspect`` and
render a self-contained ``restore.sh`` that recreates it: same image, name,
ports, environment, mounts, restart policy and original command.

Only ``shlex``/``json`` are used so this is unit-testable without Docker.
"""

from __future__ import annotations

import json
import shlex
from typing import Any


class SnapshotError(ValueError):
    pass


def _inspect_entry(inspect: Any) -> dict[str, Any]:
    """Return the single container object from ``docker inspect`` output."""
    if isinstance(inspect, str):
        inspect = json.loads(inspect)
    if isinstance(inspect, list):
        if not inspect:
            raise SnapshotError("docker inspect returned an empty array")
        entry = inspect[0]
    else:
        entry = inspect
    if not isinstance(entry, dict):
        raise SnapshotError("docker inspect entry is not an object")
    return entry


def _image_ref(entry: dict[str, Any]) -> str:
    config = entry.get("Config") or {}
    image = config.get("Image")
    if isinstance(image, str) and image:
        return image
    image = entry.get("Image")
    if isinstance(image, str) and image:
        return image
    raise SnapshotError("could not determine container image from inspect output")


def _container_name(entry: dict[str, Any]) -> str:
    name = entry.get("Name") or ""
    return name.lstrip("/") or "restored-runtime"


def _binds(entry: dict[str, Any]) -> list[str]:
    host_config = entry.get("HostConfig") or {}
    binds = host_config.get("Binds")
    if not binds:
        mounts = entry.get("Mounts") or []
        binds = []
        for mount in mounts:
            source = mount.get("Source")
            destination = mount.get("Destination")
            if source and destination:
                mode = "ro" if mount.get("RW") is False else "rw"
                binds.append(f"{source}:{destination}:{mode}")
    return [str(bind) for bind in binds or []]


def _port_flags(entry: dict[str, Any]) -> list[str]:
    host_config = entry.get("HostConfig") or {}
    bindings = host_config.get("PortBindings") or {}
    flags: list[str] = []
    for container_port, host_list in sorted(bindings.items()):
        for host in host_list or [{}]:
            host_ip = host.get("HostIp") or ""
            host_port = host.get("HostPort") or ""
            spec = f"{host_ip}:{host_port}:{container_port}" if host_ip else f"{host_port}:{container_port}"
            flags.extend(["-p", spec])
    return flags


def _env_flags(entry: dict[str, Any]) -> list[str]:
    config = entry.get("Config") or {}
    env = config.get("Env") or []
    flags: list[str] = []
    for item in env:
        if isinstance(item, str) and "=" in item:
            flags.extend(["-e", item])
    return flags


def _command(entry: dict[str, Any]) -> list[str]:
    config = entry.get("Config") or {}
    cmd = config.get("Cmd")
    return [str(part) for part in cmd] if isinstance(cmd, list) else []


def _restart_flag(entry: dict[str, Any]) -> list[str]:
    host_config = entry.get("HostConfig") or {}
    policy = (host_config.get("RestartPolicy") or {}).get("Name") or ""
    if policy and policy != "no":
        return [f"--restart={policy}"]
    return []


def _uses_gpu(entry: dict[str, Any]) -> bool:
    host_config = entry.get("HostConfig") or {}
    if host_config.get("DeviceRequests"):
        return True
    if host_config.get("Runtime") == "nvidia":
        return True
    return False


def render_restore_script(inspect: Any, *, docker_bin: str = "docker") -> str:
    """Render a ``restore.sh`` recreating the inspected container.

    ``inspect`` may be the raw ``docker inspect`` JSON string, the parsed list, or
    a single container dict. Fails closed (raises ``SnapshotError``) when the
    image cannot be determined, so a broken snapshot is never silently accepted.
    """
    entry = _inspect_entry(inspect)
    image = _image_ref(entry)
    name = _container_name(entry)

    run: list[str] = [docker_bin, "run", "-d", "--name", name]
    if _uses_gpu(entry):
        run.append("--gpus=all")
    run.extend(_restart_flag(entry))
    run.extend(_port_flags(entry))
    run.extend(_env_flags(entry))
    for bind in _binds(entry):
        run.extend(["-v", bind])
    run.append(image)
    run.extend(_command(entry))

    quoted = " ".join(shlex.quote(part) for part in run)
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Auto-generated runtime restore artifact.",
        f"# Recreates container {name!r} from image {image!r} captured via docker inspect.",
        f"# Remove any conflicting container first: {docker_bin} rm -f {shlex.quote(name)}",
        f"{docker_bin} rm -f {shlex.quote(name)} 2>/dev/null || true",
        quoted,
        "",
    ]
    return "\n".join(lines)
