"""Public-export transforms keep public non-host paths intact (PR #96 review).

The generic absolute-path rewrite in the export sanitizer turned two kinds of public paths into the
`${PUBLIC_WORKSPACE}` placeholder: an HTTP API route (`/v1/chat/completions`) and the kernel's descriptor
directories (`/proc/self/fd`, including `f"/proc/self/fd/{fd}"`). Neither names operator host layout, and the rewrite
broke the exported clients and validators. Host paths next to them, and route- or descriptor-shaped host paths, must
still be rewritten. Compose files get no container-path exemption: a default that must survive export is written as
a two-segment path (`/work/current-run`), which the rewrite does not touch.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from model_forge.public_export.transforms import (
    _PUBLIC_SYSTEM_PATH_PREFIXES,
    TransformContext,
    apply_transform,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# A host identity root, assembled so the exporter's detector does not read the fixture itself as a leak.
HOME_ROOT = "/" + "home"


def _context(path: str) -> TransformContext:
    return TransformContext(source_path=path, source_sha="0" * 40, public_contact="security@example.com")


def test_python_script_keeps_api_routes_and_kernel_paths() -> None:
    source = (
        b'c.request("POST", "/v1/chat/completions", body=b"")\n'
        b'FD_DIR = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"\n'
        b'CACHE = "/mnt/d/model-forge/cache/export"\n'
    )
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/client.py")).data.decode()
    assert 'c.request("POST", "/v1/chat/completions", body=b"")' in out
    assert 'FD_DIR = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"' in out
    assert "/mnt/d/model-forge" not in out


def test_route_shaped_host_paths_are_still_rewritten() -> None:
    source = b'A = "/v1/home/operator/secrets/token"\nB = "/v2/Users/operator/cache/x"\n'
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/client.py")).data.decode()
    assert "operator" not in out


def test_live_omni_client_and_build_compose_survive_export() -> None:
    path = "models/nemotron-3-nano-omni-r1/eval/omni_check.py"
    source = (REPO_ROOT / path).read_text()
    out = apply_transform("sanitize_python_script", source.encode(), _context(path)).data.decode()
    # Whatever form the client spells its route in (a literal, or the two-part constant main uses), every line that
    # carries it survives export verbatim.
    route_lines = [line for line in source.splitlines() if "/v1/" in line]
    assert route_lines
    for line in route_lines:
        assert line in out.splitlines()
    assert "${PUBLIC_WORKSPACE}" not in out
    path = "containers/build/docker-compose.yml"
    out = apply_transform("sanitize_and_validate_compose", (REPO_ROOT / path).read_bytes(), _context(path)).data.decode()
    assert "RUN_ROOT=${RUN_ROOT:-/work/current-run}" in out
    assert "${PUBLIC_" not in out


def test_kernel_and_route_exemptions_do_not_pass_host_paths() -> None:
    """Only the descriptor directories themselves are kept; `/proc/self/root/...` and `..` segments resolve into the
    host filesystem, so they are rewritten like any other host path (PR #96 review round 2)."""
    source = (
        b'A = "/proc/self/root/home/operator/secret/key"\n'
        b'B = "/proc/self/cwd/operator/data/file"\n'
        b'C = "/v1/../srv/operator/data/x"\n'
        b'D = "/dev/fd/3"\n'
        b'E = "/proc/self/fd/7"\n'
    )
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/client.py")).data.decode()
    assert "operator" not in out
    assert 'D = "/dev/fd/3"' in out
    assert 'E = "/proc/self/fd/7"' in out


def test_interpolated_descriptor_paths_survive_export() -> None:
    """The live code re-opens held descriptors through f-strings and str.format (PR #96 Bugbot, round 3). Exactly one
    interpolated or numeric entry under a descriptor directory is kept; anything deeper is a host path."""
    source = (
        b'A = f"/proc/self/fd/{fd}"\n'
        b'B = f"/proc/self/fd/{self._fd}"\n'
        b'C = "/proc/self/fd/{}".format(fd)\n'
        b'D = f"/proc/thread-self/fd/{dir_fd}"\n'
        b'E = f"/proc/self/fd/{fd}/../../root/home/operator/key"\n'
        b'F = "/dev/fd/{x}/operator"\n'
    )
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/client.py")).data.decode()
    for kept in ('A = f"/proc/self/fd/{fd}"', 'B = f"/proc/self/fd/{self._fd}"', 'C = "/proc/self/fd/{}".format(fd)',
                 'D = f"/proc/thread-self/fd/{dir_fd}"'):
        assert kept in out
    assert "operator" not in out
    shell = b'exec 3<"$IN"\nREAD_FROM=/dev/fd/$FD\nALSO=/dev/fd/${FD}\nNOT=/dev/fd/${FD}/operator/x\n'
    out = apply_transform("sanitize_container_script", shell, _context("scripts/x/run.sh")).data.decode()
    assert "READ_FROM=/dev/fd/$FD" in out and "ALSO=/dev/fd/${FD}" in out
    assert "operator" not in out


def test_live_descriptor_reopen_survives_export() -> None:
    path = "src/model_forge/pipeline.py"
    source = (REPO_ROOT / path).read_bytes()
    assert b'f"/proc/self/fd/{fd}"' in source
    out = apply_transform("sanitize_python_script", source, _context(path)).data.decode()
    assert 'f"/proc/self/fd/{fd}"' in out and 'f"/proc/self/fd/{self._fd}"' in out
    assert "${PUBLIC_WORKSPACE}" not in out


@pytest.mark.parametrize(
    "value",
    # Only 3+-segment paths reach the rewrite at all (`/v1/home` is left alone, as on main).
    ["/v1/Volumes/operator/x", "/v2/Users/operator/cache", "/v1/x/Volumes"],
)
def test_routes_containing_a_host_identity_root_are_rewritten(value: str) -> None:
    out = apply_transform("sanitize_python_script", f'A = "{value}"\n'.encode(), _context("models/x/eval/c.py")).data.decode()
    assert value not in out


@pytest.mark.parametrize(
    "line",
    [
        # A path under a declared bind target or named-volume target is still a host path: declaring the mount
        # exempts nothing under it (each case would pass a reinstated container-root exemption only by leaking).
        "      - '${NAME:-.}:/datasets:ro'\n    environment:\n      - DS=/datasets/operator/x",
        "      - {type: bind, source: data, target: /data}\n    environment:\n      - OUT=/data/operator/runs",
        "    env_file: /work/operator/secrets.env",
        "      - OUT=${OUT:-/work/operator/runs}",
    ],
)
def test_compose_has_no_container_path_exemption(line: str) -> None:
    """No path in a compose file survives on the strength of a declared mount (PR #96 review round 4): host-side
    bind sources, env files and defaults under a named volume are all rewritten."""
    source = ("services:\n  runner:\n    volumes:\n      - work-state:/work\n" + line + "\n").encode()
    out = apply_transform("sanitize_and_validate_compose", source, _context("containers/build/docker-compose.yml")).data.decode()
    assert "operator" not in out


@pytest.mark.parametrize(
    "value",
    ["/v1/chat/completions/../../srv/operator/x", "/v1/chat/.hidden/operator/x"],
)
def test_a_route_prefix_does_not_exempt_the_rest_of_the_path(value: str) -> None:
    """The whole value must be a route; a route-shaped prefix followed by a dotted segment is a host path."""
    source = f'A = "{value}"\n'.encode()
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/c.py")).data.decode()
    assert "operator" not in out


@pytest.mark.parametrize(
    ("line", "kept"),
    [
        ("      - ROUTE=${ROUTE:-/v1/chat/completions}", "ROUTE=${ROUTE:-/v1/chat/completions}"),
        ("      - FD=${FD_PATH:-/proc/self/fd/7}", "FD=${FD_PATH:-/proc/self/fd/7}"),
    ],
)
def test_exemptions_hold_inside_a_compose_default(line: str, kept: str) -> None:
    """The closing brace of `${NAME:-...}` is matched together with the path; the exemption must still see the path
    alone, and the brace must stay outside it (the compose transform does not rewrite parameter defaults first)."""
    source = ("services:\n  runner:\n    environment:\n" + line + "\n").encode()
    out = apply_transform("sanitize_and_validate_compose", source, _context("containers/build/docker-compose.yml")).data.decode()
    assert kept in out


@pytest.mark.parametrize(
    "line",
    [
        'OUT="${_out_dir:-/data/operator/runs}"',  # name starting with `_`
        'OUT="${OUT2:-/data/operator/runs}"',      # digit in the name
        'OUT="${10:-/data/operator/runs}"',        # positional parameter with two digits
        'OUT="${3:-/data/operator/runs}"',         # positional parameter with one digit
    ],
)
def test_every_shell_default_name_form_requires_an_argument(line: str) -> None:
    """Each parameter-name form a shell accepts must hit the required-argument rewrite (PR #96 review round 5)."""
    source = f"#!/bin/bash\n{line}\n".encode()
    out = apply_transform("sanitize_container_script", source, _context("scripts/x/run.sh")).data.decode()
    assert "operator" not in out
    assert ":?" in out


def test_positional_and_named_defaults_get_their_own_message() -> None:
    source = b'#!/bin/bash\nA="${1:-/data/operator/a}"\nB="${OUT:-/data/operator/b}"\n'
    out = apply_transform("sanitize_container_script", source, _context("scripts/x/run.sh")).data.decode()
    assert "${1:?Pass argument 1 as the public artifact path}" in out
    assert "${OUT:?Set OUT to the public artifact path}" in out


@pytest.mark.parametrize("value", ["/v/srv/operator/x", "/vault/srv/operator/x", "/v1x/srv/operator/x"])
def test_only_a_numbered_version_prefix_is_a_route(value: str) -> None:
    out = apply_transform("sanitize_python_script", f'A = "{value}"\n'.encode(), _context("models/x/eval/c.py")).data.decode()
    assert "operator" not in out


def test_routes_with_interpolated_segments_are_kept() -> None:
    source = (b'URL = f"{base}/v1/models/{model_id}/score"\nU2 = f"/v2/runs/${run}/x"\n'
              b'U3 = f"/v1/models/{model_id}/files"\nU4 = "/v1/models/abc/files"\n')
    out = apply_transform("sanitize_python_script", source, _context("models/x/eval/c.py")).data.decode()
    assert 'f"{base}/v1/models/{model_id}/score"' in out
    assert 'f"/v2/runs/${run}/x"' in out
    # A route that starts the string reaches the exemption itself (the `{base}` case above never does): an
    # interpolated segment and a plain `models` segment both stay a route, not an artifact path.
    assert 'U3 = f"/v1/models/{model_id}/files"' in out
    assert 'U4 = "/v1/models/abc/files"' in out


@pytest.mark.parametrize(
    ("transform", "path", "line"),
    [
        # Only `v` + digits is a version prefix: any other letter before the digits is a host path.
        ("sanitize_python_script", "models/x/eval/c.py", 'A = "/u01/operator/data"'),
        ("sanitize_public_markdown", "docs/x.md", "runs under /d2/operator/runs/x"),
        # A `$NAME` descriptor entry is the end of the path: a child after it is a host path, in every file kind.
        ("sanitize_container_script", "scripts/x/run.sh", "#!/bin/bash\nX=/dev/fd/$FD/operator/secret"),
        ("sanitize_python_script", "models/x/eval/c.py", 'A = "/dev/fd/$FD/operator/secret"'),
        ("sanitize_public_markdown", "docs/x.md", "X=/dev/fd/$FD/operator/secret"),
        # `{...}` and `${...}` entries hold one identifier, never a path.
        ("sanitize_python_script", "models/x/eval/c.py", 'A = "/dev/fd/{/srv/operator/x}"'),
        ("sanitize_python_script", "models/x/eval/c.py", 'A = "/proc/self/fd/${/srv/operator/x}"'),
    ],
)
def test_route_and_descriptor_look_alikes_are_rewritten(transform: str, path: str, line: str) -> None:
    """Each loosening of the route and descriptor exemptions published one of these host paths (PR #96 review
    round 6: four mutants passed the suite)."""
    out = apply_transform(transform, f"{line}\n".encode(), _context(path)).data.decode()
    assert "operator" not in out


@pytest.mark.parametrize("prefix", _PUBLIC_SYSTEM_PATH_PREFIXES)
def test_every_public_system_prefix_is_exact(prefix: str) -> None:
    """A look-alike of each public system prefix (the prefix without its final slash, then more letters) is a host
    path: `/binaries/...` is not under `/bin/` (PR #96 review round 6: 17 trailing-slash mutants passed the suite)."""
    value = prefix.rstrip("/") + "zz/operator/x/y"
    out = apply_transform("sanitize_public_markdown", f"path = {value}\n".encode(), _context("docs/x.md")).data.decode()
    assert "operator" not in out


@pytest.mark.parametrize(
    "value",
    [
        "/proc/self/fd/{fd}/{name}",  # a second entry under a descriptor
        "/proc/self/fd/3/4",          # two numbered entries
        "/proc/self/fd/cwd/operator", # a named entry with a child
        "/dev/fd/stdin/operator",     # a named entry under /dev/fd, with a child
        "/proc/self/fd/operator",     # a named single entry: descriptors are numbers, not names
        "/dev/fd/operator",           # the same under /dev/fd
    ],
)
def test_a_descriptor_path_has_at_most_one_numeric_or_interpolated_entry(value: str) -> None:
    out = apply_transform("sanitize_python_script", f'A = f"{value}"\n'.encode(), _context("models/x/eval/c.py")).data.decode()
    assert value not in out


@pytest.mark.parametrize(
    "line",
    [
        'X="${A:-${B:-/srv/operator/runs/x}}"',    # nested default: two closing braces
        'X="${A:-/srv/operator/models/x}"',         # artifact placeholder branch
        'X="${A:-/srv/operator/workspace/x}"',      # private-words placeholder branch
        'X="${A:-/usr/local/bin/tool}"',            # public system prefix: kept, brace kept
        'X="${A:-/srv/operator/plain/x}"',          # final placeholder branch
    ],
)
def test_every_rewrite_branch_keeps_the_expansion_balanced(line: str, tmp_path: Path) -> None:
    """Whatever branch rewrites (or keeps) the path, the `${...}` expansion stays balanced and the line still parses
    as shell (PR #96 review round 5: four brace mutants passed)."""
    out = apply_transform("sanitize_public_markdown", f"{line}\n".encode(), _context("docs/x.md")).data.decode()
    assert "operator" not in out
    assert out.count("{") == out.count("}"), out
    script = tmp_path / "x.sh"
    script.write_text(out)
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "line",
    [
        "results_dir=/srv/operator/runs/out",     # `re` + letters is not the `re:` marker
        "path = /library/operator/x/y",           # /library is not the public /lib/ prefix
    ],
)
def test_pre_existing_exemptions_are_exact(line: str) -> None:
    """Two guards that predate this PR (the `re:` pattern marker, the `/lib/` system prefix) keep only their exact
    forms; a prefix-shaped look-alike is a host path (PR #96 review round 5 suggestions)."""
    out = apply_transform("sanitize_public_markdown", f"{line}\n".encode(), _context("docs/x.md")).data.decode()
    assert "operator" not in out
