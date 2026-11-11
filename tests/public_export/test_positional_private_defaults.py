"""Public export must not keep private absolute paths in sanitized scripts.

Positional and non-uppercase shell defaults such as ``${1:-/mnt/<drive>/...}`` are
private host paths. Every text transform has to drop them, and the content
scanner has to reject them when a transform does not. These tests fail while
that gap is open.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

from model_forge.public_export.detectors import (
    DetectorPolicy,
    GitleaksStatus,
    scan_file,
)
from model_forge.public_export.exporter import (
    GITLEAKS_SCOPE,
    ExportError,
    ExportRequest,
    GitleaksEvidence,
    _load_manifest,
    _Rule,
    export_public,
)
from model_forge.public_export.transforms import (
    TransformContext,
    TransformError,
    apply_transform,
)
from model_forge.public_export.verifier import PublicVerifyError, _scan_metadata

REPO_ROOT = Path(__file__).resolve().parents[2]
LIGHTNING_SCRIPT = "scripts/nemotron-3.5_lightning/post_process_ablit_nvfp4.sh"
PUBLIC_CONTACT = "security@hangglidersrule.com"
SOURCE_SHA = "a" * 40

# Absolute paths the sanitizer claims to rewrite, written so the character
# before the slash is the dash in ``:-``.
_DASH_ABSOLUTE_PATHS = (
    "/mnt/d/model-forge/cache/export",
    "/mnt/e/other-workspace/private/build",
    "/opt/model-forge/artifacts/model",
    "/srv/team/checkpoints/run",
    "/tmp/acme/workspace/build",
    "/var/lib/model-forge/weights/model",
)


def _context(path: str) -> TransformContext:
    return TransformContext(
        source_path=path,
        source_sha=SOURCE_SHA,
        public_contact="security@example.com",
    )


def _assert_absent(data: bytes, needles: tuple[str, ...]) -> None:
    text = data.decode("utf-8")
    for needle in needles:
        assert needle not in text


def _dash_text(prefix: str, suffix: str = "") -> bytes:
    lines = [f"{prefix}{path}{suffix}" for path in _DASH_ABSOLUTE_PATHS]
    return ("".join(f"{line}\n" for line in lines)).encode("utf-8")


def _scan(tmp_path: Path, relative: str, text: str) -> set[str]:
    root = tmp_path / "root"
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    findings = scan_file(
        root,
        relative,
        policy=DetectorPolicy(
            max_file_bytes=8192,
            max_scan_bytes=8192,
            public_contacts=frozenset({PUBLIC_CONTACT}),
        ),
        gitleaks_status=GitleaksStatus.PASSED,
    )
    return {finding.rule_id for finding in findings}


class _PassingGitleaks:
    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        assert source.is_dir()
        return GitleaksEvidence(
            version="8.30.1",
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
            scope=GITLEAKS_SCOPE,
            source_sha=source_sha,
        )


def _rule(
    source: str,
    *,
    disposition: str = "copy",
    transformation: str | None = None,
) -> dict[str, object]:
    return {
        "id": source.replace("/", "-").replace("*", "all"),
        "source": source,
        "disposition": disposition,
        "public_destination": "{source}",
        "reason": "Synthetic publication rule for private-default coverage.",
        "transformation": transformation,
        "owner": "tests",
        "max_size_bytes": 131072,
        "generated": False,
        "precedence": 100,
    }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(tmp_path: Path, files: dict[str, bytes], rules: list[dict[str, object]]) -> tuple[Path, Path, str]:
    repo = tmp_path / "private-source"
    repo.mkdir()
    for relative, data in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    manifest = repo / "public-files.yaml"
    manifest.write_text(
        yaml.safe_dump({"version": 1, "rules": rules}, sort_keys=False),
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Exporter Tests",
        "-c",
        "user.email=exporter@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return repo, manifest, _git(repo, "rev-parse", "HEAD")


def _export(repo: Path, manifest: Path, sha: str, output: Path) -> None:
    export_public(
        ExportRequest(
            source=repo,
            output=output,
            manifest=manifest,
            source_sha=sha,
            gitleaks_runner=_PassingGitleaks(),
            public_contact=PUBLIC_CONTACT,
        )
    )


def _promoted_bytes(output: Path) -> list[tuple[str, bytes]]:
    if not output.exists():
        return []
    promoted: list[tuple[str, bytes]] = []
    for path in output.rglob("*"):
        if path.is_file():
            promoted.append((path.relative_to(output).as_posix(), path.read_bytes()))
    return promoted


def _winning_rule(rules: list[_Rule], path: str) -> _Rule:
    candidates = [rule for rule in rules if rule.pattern.fullmatch(path)]
    assert candidates
    candidates.sort(key=lambda rule: rule.precedence, reverse=True)
    winner = candidates[0]
    if len(candidates) > 1:
        assert winner.precedence > candidates[1].precedence
        unresolved = {rule.rule_id for rule in candidates[1:]} - winner.resolves
        assert not unresolved
    return winner


def test_lightning_post_process_script_is_published_with_container_sanitizer() -> None:
    """The reported script is classified for publication, not excluded."""

    manifest = (REPO_ROOT / "tools/public_export/public-files.yaml").read_bytes()
    rule = _winning_rule(_load_manifest(manifest), LIGHTNING_SCRIPT)

    assert rule.disposition == "transform"
    assert rule.transformation == "sanitize_container_script"
    assert rule.destination == "{source}"
    assert rule.detector_suppressions == frozenset()


@pytest.mark.parametrize(
    "transform",
    ["sanitize_container_script", "sanitize_qwen_script"],
)
@pytest.mark.parametrize(
    ("source", "needle"),
    [
        (
            "EXPORT=${1:-/mnt/d/model-forge/cache/lightning-ablit-export}\n",
            "/mnt/d/model-forge/cache/lightning-ablit-export",
        ),
        (
            "SRC_BF16=${2:-/mnt/d/model-forge/ablit/runs/apply_abliteration}\n",
            "/mnt/d/model-forge/ablit/runs/apply_abliteration",
        ),
        (
            'CFG="${3:-/mnt/d/model-forge/lightning-bf16}"\n',
            "/mnt/d/model-forge/lightning-bf16",
        ),
        (
            'EXPORT="${10:-/mnt/e/other-workspace/private/build}"\n',
            "/mnt/e/other-workspace/private/build",
        ),
        (
            "export_dir=${export_dir:-/opt/model-forge/artifacts/model}\n",
            "/opt/model-forge/artifacts/model",
        ),
        (
            "ROOT='${artifact:-/srv/team/checkpoints/run}'\n",
            "/srv/team/checkpoints/run",
        ),
        (
            "cache=${cache:-/tmp/acme/workspace/build}\n",
            "/tmp/acme/workspace/build",
        ),
        (
            "weights=${weights:-/var/lib/model-forge/weights/model}\n",
            "/var/lib/model-forge/weights/model",
        ),
    ],
)
def test_shell_transforms_strip_positional_and_lowercase_private_defaults(
    tmp_path: Path,
    transform: str,
    source: str,
    needle: str,
) -> None:
    assert needle in source
    result = apply_transform(transform, source.encode(), _context("scripts/launch.sh"))
    script = tmp_path / "launch.sh"
    script.write_bytes(result.data)

    assert needle not in result.data.decode("utf-8")
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_python_script_transform_strips_positional_private_defaults() -> None:
    source = (
        "def main() -> None:\n"
        '    command = "${1:-/mnt/e/other-workspace/private/build}"\n'
        '    other = "${export_dir:-/tmp/acme/workspace/build}"\n'
    )

    result = apply_transform(
        "sanitize_python_script",
        source.encode(),
        _context("scripts/tool.py"),
    )

    _assert_absent(result.data, _DASH_ABSOLUTE_PATHS)


def test_dockerfile_branch_strips_positional_private_defaults() -> None:
    source = (
        "FROM example\n"
        "ENV EXPORT=${1:-/mnt/d/model-forge/cache/lightning-ablit-export}\n"
        "ENV SRC=${export_dir:-/opt/model-forge/artifacts/model}\n"
    )

    result = apply_transform(
        "sanitize_container_script",
        source.encode(),
        _context("containers/modelopt/Dockerfile"),
    )

    _assert_absent(result.data, _DASH_ABSOLUTE_PATHS)
    assert b"FROM example\n" in result.data


@pytest.mark.parametrize(
    ("transform", "source_path", "payload"),
    [
        (
            "sanitize_public_markdown",
            "docs/guide.md",
            _dash_text("see prefix:-"),
        ),
        (
            "sanitize_public_model_card",
            "models/family/card.md",
            _dash_text("see ${1:-"),
        ),
        (
            "sanitize_validation_inventory",
            "docs/inventory.txt",
            _dash_text("value ${2:-"),
        ),
        (
            "sanitize_public_gitignore",
            ".gitignore",
            _dash_text("prefix:-"),
        ),
        (
            "sanitize_and_validate_modelopt_config",
            "configs/modelopt/recipe.yaml",
            _dash_text('note: "prefix:-', '"'),
        ),
        (
            "sanitize_and_validate_compose",
            "containers/docker-compose.yml",
            _dash_text('note: "prefix:-', '"'),
        ),
        (
            "sanitize_and_validate_serve_profile",
            "containers/serve/profile.yml",
            _dash_text('note: "prefix:-', '"'),
        ),
        (
            "sanitize_serving_capacity_profiles",
            "profiles/capacity.json",
            b'{"note": "prefix:-/mnt/d/model-forge/cache/export"}\n',
        ),
        (
            "sanitize_artifact_manifest",
            "artifacts/manifest.yaml",
            _dash_text('note: "prefix:-', '"'),
        ),
    ],
)
def test_text_transforms_strip_absolute_paths_after_a_dash(
    transform: str,
    source_path: str,
    payload: bytes,
) -> None:
    result = apply_transform(transform, payload, _context(source_path))

    _assert_absent(result.data, _DASH_ABSOLUTE_PATHS)


def test_recipe_transform_rejects_dash_preceded_private_paths() -> None:
    payload = b"schema_version: '2.0'\n" + _dash_text('note: "prefix:-', '"')

    with pytest.raises(TransformError, match="private runtime content"):
        apply_transform(
            "sanitize_and_validate_recipe",
            payload,
            _context("recipes/private.yaml"),
        )


@pytest.mark.private_source_only
def test_live_lightning_script_loses_wsl_defaults(tmp_path: Path) -> None:
    # Reads the PRIVATE source of the script, whose host defaults this transform removes; the exported copy no longer
    # carries them (that is the fix), so this runs only on the private tree.
    source = (REPO_ROOT / LIGHTNING_SCRIPT).read_bytes()
    assert b"/mnt/d/model-forge/cache/lightning-ablit-export" in source
    assert b"/mnt/d/model-forge/ablit/runs/apply_abliteration" in source
    assert b"/mnt/d/model-forge/lightning-bf16" in source

    result = apply_transform(
        "sanitize_container_script",
        source,
        _context(LIGHTNING_SCRIPT),
    )
    script = tmp_path / "post_process_ablit_nvfp4.sh"
    script.write_bytes(result.data)

    _assert_absent(
        result.data,
        (
            "/mnt/d/model-forge/cache/lightning-ablit-export",
            "/mnt/d/model-forge/ablit/runs/apply_abliteration",
            "/mnt/d/model-forge/lightning-bf16",
        ),
    )
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


@pytest.mark.parametrize(
    "text",
    [
        "EXPORT=${1:-/mnt/d/model-forge/cache/lightning-ablit-export}\n",
        "SRC_BF16=${2:-/mnt/d/model-forge/ablit/runs/apply_abliteration}\n",
        "SRC_BF16_CFG=${3:-/mnt/d/model-forge/lightning-bf16}\n",
        'EXPORT="${10:-/mnt/e/other-workspace/private/build}"\n',
        "export_dir=${export_dir:-/opt/model-forge/artifacts/model}\n",
        "see prefix:-/srv/team/checkpoints/run\n",
        "see prefix:-/tmp/acme/workspace/build\n",
        "see prefix:-/var/lib/model-forge/weights/model\n",
    ],
)
def test_detector_flags_private_absolute_paths_left_by_the_sanitizer(
    tmp_path: Path,
    text: str,
) -> None:
    rules = _scan(tmp_path, "scripts/launch.sh", text)

    assert "path.absolute-posix" in rules


@pytest.mark.parametrize(
    ("relative", "text"),
    [
        # A root plus one name: outside scripts the transforms keep a default this short, so the detector is the
        # only guard (PR #96 review round 6: `{2,}` instead of `+` passed the suite).
        ("docs/run.md", "OUT=${OUT:-/srv/operator}\n"),
        ("containers/x/docker-compose.yml", "      - OUT=${OUT:-/srv/operator}\n"),
        # A quoted default, double and single (the quote before the path was optional only in the transform tests).
        ("docs/run.md", 'OUT=${OUT:-"/srv/operator"}\n'),
        ("containers/x/docker-compose.yml", "      - OUT=${OUT:-'/var/operator'}\n"),
    ],
)
def test_detector_flags_short_and_quoted_host_defaults(tmp_path: Path, relative: str, text: str) -> None:
    assert "path.absolute-posix" in _scan(tmp_path, relative, text)


@pytest.mark.parametrize(
    "text",
    [
        "container=/mnt/export\n",
        "home=/mnt/hf_cache\n",
        "python=/usr/bin/env\n",
        "shell=/bin/sh\n",
        "brew=/opt/homebrew/bin/python3\n",
        "PY=${PY:-/opt/homebrew/bin/python3}\n",  # a Homebrew default is public, not a host /opt default
        'case "${ROOT}" in /mnt/d/*|/d/*) ;; *) exit 1 ;; esac\n',
        "endpoint: /api/v1/models\n",
    ],
)
def test_detector_ignores_container_mounts_and_system_paths(tmp_path: Path, text: str) -> None:
    assert _scan(tmp_path, "scripts/launch.sh", text).isdisjoint(
        {"path.absolute-posix", "path.absolute-windows"}
    )


def test_verifier_metadata_scan_rejects_unsanitized_wsl_defaults(tmp_path: Path) -> None:
    root = tmp_path / "public"
    relative = "scripts/post_process.sh"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.write_text(
        "EXPORT=${1:-/mnt/d/model-forge/cache/lightning-ablit-export}\n",
        encoding="utf-8",
    )
    (root / "PUBLIC_EXPORT_MANIFEST.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicVerifyError, match="path.absolute-posix"):
        _scan_metadata(root, {relative}, PUBLIC_CONTACT, frozenset())


def test_export_does_not_publish_lightning_or_future_positional_defaults(tmp_path: Path) -> None:
    live = (REPO_ROOT / LIGHTNING_SCRIPT).read_bytes()
    future = (
        "#!/bin/sh\n"
        "EXPORT=${1:-/mnt/e/other-workspace/private/build}\n"
        "SRC=${export_dir:-/opt/model-forge/artifacts/model}\n"
        "CFG=${2:-/srv/team/checkpoints/run}\n"
    ).encode()
    future_path = "scripts/nemotron-3.5_lightning/future_defaults.sh"
    files = {LIGHTNING_SCRIPT: live, future_path: future}
    repo, manifest, sha = _commit(
        tmp_path,
        files,
        [
            _rule("public-files.yaml"),
            _rule(
                "scripts/nemotron-3.5_lightning/**",
                disposition="transform",
                transformation="sanitize_container_script",
            ),
        ],
    )
    output = tmp_path / "public"
    needles = (
        "/mnt/d/model-forge",
        "/mnt/e/other-workspace",
        "/opt/model-forge/artifacts",
        "/srv/team/checkpoints",
    )

    try:
        _export(repo, manifest, sha, output)
    except ExportError as error:
        assert "path.absolute-posix" in str(error)
        for _relative, data in _promoted_bytes(output):
            _assert_absent(data, needles)
        return

    published = {relative: data for relative, data in _promoted_bytes(output)}
    assert LIGHTNING_SCRIPT in published
    assert future_path in published
    for data in published.values():
        _assert_absent(data, needles)


def test_copy_export_does_not_promote_a_wsl_host_path(tmp_path: Path) -> None:
    relative = "docs/operator-note.md"
    files = {
        relative: b"cache prefix:-/mnt/d/model-forge/runtime-cache/mf-tune\n",
    }
    repo, manifest, sha = _commit(
        tmp_path,
        files,
        [_rule("public-files.yaml"), _rule(relative)],
    )
    output = tmp_path / "public"

    with pytest.raises(ExportError, match="path.absolute-posix"):
        _export(repo, manifest, sha, output)

    for _relative, data in _promoted_bytes(output):
        assert b"/mnt/d/model-forge" not in data


# Fragments joined at run time, so no private-looking path sits in this file as a literal.
_HOST_ROOTS = ("/" + "mnt", "/" + "media", "/" + "srv")


@pytest.mark.parametrize("root", _HOST_ROOTS)
@pytest.mark.parametrize("operator", [":-", "-", ":=", "="])
@pytest.mark.parametrize("name", ["1", "export_dir"])
def test_detector_flags_every_default_operator_and_host_root(
    tmp_path: Path, root: str, operator: str, name: str
) -> None:
    text = "X=${" + name + operator + root + "/d/model-forge/run}\n"
    assert "path.absolute-posix" in _scan(tmp_path, "scripts/launch.sh", text)


@pytest.mark.parametrize("home", ["/" + "Users", "/" + "home", "/" + "Volumes"])
def test_detector_flags_a_home_or_volume_path_after_a_dash(tmp_path: Path, home: str) -> None:
    text = "see prefix:-" + home + "/someone/model-forge\n"
    assert "path.absolute-posix" in _scan(tmp_path, "scripts/launch.sh", text)


@pytest.mark.parametrize(
    "transform",
    ["sanitize_container_script", "sanitize_qwen_script"],
)
@pytest.mark.parametrize("operator", [":-", "-", ":=", "="])
@pytest.mark.parametrize("name", ["1", "export_dir", "EXPORT_DIR"])
@pytest.mark.parametrize("quote", ["", '"', "'"])
def test_shell_transforms_make_every_absolute_default_a_required_argument(
    tmp_path: Path, transform: str, operator: str, name: str, quote: str
) -> None:
    value = "/" + "mnt/d/model-forge/cache/export"
    source = f'EXPORT="${{{name}{operator}{quote}{value}{quote}}}"\n'
    result = apply_transform(transform, source.encode(), _context("scripts/launch.sh"))
    text = result.data.decode("utf-8")
    assert f"${{{name}:?" in text, text
    assert "model-forge/cache" not in text, text
    script = tmp_path / "launch.sh"
    script.write_bytes(result.data)
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)


def test_path_placeholder_keeps_the_expansion_brace_outside(tmp_path: Path) -> None:
    """A path ending a ${...} expansion is replaced without swallowing its closing brace."""
    source = 'echo "${HOME_ROOT:+' + "/" + 'opt/acme/workspace/build}"\n'
    result = apply_transform("sanitize_container_script", source.encode(), _context("scripts/launch.sh"))
    text = result.data.decode("utf-8")
    assert "acme/workspace" not in text, text
    assert text.count("{") == text.count("}"), text
    script = tmp_path / "launch.sh"
    script.write_bytes(result.data)
    subprocess.run(["bash", "-n", str(script)], check=True, capture_output=True)
