"""Spec for the public-export test_ fixture exemption.

The exemption must apply only under tests/, and only to the span of a string or
bytes constant. These tests fail while a function whose name starts with test_
suppresses contextual findings for the whole line in any Python file.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
import yaml

import model_forge.public_export.exporter as exporter_module
from model_forge.public_export.detectors import (
    RULE_CGNAT_IPV4,
    RULE_COLLABORATION_METADATA,
    RULE_FLEET_HOSTNAME,
    RULE_NONPUBLIC_EMAIL,
    RULE_POSIX_PATH,
    RULE_PRIVATE_IPV4,
    RULE_RAW_BENCHMARK_KEY,
    RULE_SESSION_METADATA,
    RULE_WINDOWS_PATH,
    DetectorPolicy,
    GitleaksStatus,
    scan_file,
)
from model_forge.public_export.exporter import (
    DEFAULT_PUBLIC_CONTACT,
    ExportError,
    ExportRequest,
    GitleaksEvidence,
    export_public,
)
from model_forge.public_export.verifier import (
    FilesystemGitleaksEvidence,
    PublicVerifyError,
    PublicVerifyRequest,
    verify_public_export,
)

FLEET_HOSTNAMES = frozenset({"forge-worker-07.private"})
POLICY = DetectorPolicy(public_contacts=frozenset({DEFAULT_PUBLIC_CONTACT}))
SRC_RELATIVE = "src/model_forge/public_surface.py"
PYI_RELATIVE = "src/model_forge/public_surface.pyi"
SRC_TESTS_RELATIVE = "src/tests/published.py"
TESTS_COMMENT_RELATIVE = "tests/public_export/test_fixture_boundary.py"
TESTS_LITERAL_RELATIVE = "tests/public_export/test_fixture_literals.py"

# benchmark.raw-key is an approved manifest suppression for copied src/tests
# Python, and verify drops that same rule for those paths. The other rules are
# not approved suppressions.
SRC_SCAN_RULES = frozenset(
    {
        RULE_PRIVATE_IPV4,
        RULE_CGNAT_IPV4,
        RULE_POSIX_PATH,
        RULE_WINDOWS_PATH,
        RULE_FLEET_HOSTNAME,
        RULE_NONPUBLIC_EMAIL,
        RULE_SESSION_METADATA,
        RULE_COLLABORATION_METADATA,
        RULE_RAW_BENCHMARK_KEY,
    }
)
SRC_PUBLISH_RULES = SRC_SCAN_RULES - {RULE_RAW_BENCHMARK_KEY}
TESTS_COMMENT_RULES = frozenset(
    {
        RULE_PRIVATE_IPV4,
        RULE_CGNAT_IPV4,
        RULE_POSIX_PATH,
        RULE_WINDOWS_PATH,
        RULE_NONPUBLIC_EMAIL,
    }
)

# Each rule appears in exactly one syntactic position. A fix that still ignores
# methods, async functions, nested functions, bytes constants, or a comment
# beside a constant drops that rule.
SRC_SOURCE = """\
class Publisher:
    def test_method(self) -> None:
        endpoint = b"10.1.2.3"


async def test_async() -> None:
    endpoint = "100.64.1.2"


def helper() -> None:
    def test_nested() -> None:
        home = "/home/operator/models/release"


def test_publish() -> None:
    windows = "C:/Users/operator/models"
    host = "forge-worker-07.private"
    contact = "operator@company.internal"
    meta = "session_id: placeholder"
    benchmark = {"question": "hidden"}
    marker = "safe"  # https://app.slack.com/client/T000/C000
"""

PYI_SOURCE = """\
def test_publish() -> None:
    endpoint = "10.1.2.3"
"""

SRC_TESTS_SOURCE = """\
def test_publish() -> None:
    endpoint = "172.16.4.5"
"""

TESTS_COMMENT_SOURCE = """\
def test_fixture() -> None:
    marker = "safe"  # 10.1.2.3
    raw = b"safe"  # /home/operator/models/release
    label = "safe"  # C:/Users/operator/models
    note = "safe"  # operator@company.internal
    block = \"\"\"safe
\"\"\"  # 100.64.1.2
"""

TESTS_LITERAL_SOURCE = """\
def test_fixture() -> None:
    marker = "10.1.2.3"
    raw = b"/home/operator/models/release"
    windows = "C:/Users/operator/models"
    block = \"\"\"100.64.1.2
\"\"\"
"""


def _rule_ids(root: Path, relative: str) -> set[str]:
    findings = scan_file(
        root,
        relative,
        policy=POLICY,
        fleet_hostnames=FLEET_HOSTNAMES,
        gitleaks_status=GitleaksStatus.PASSED,
    )
    return {finding.rule_id for finding in findings}


def _write(root: Path, relative: str, text: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rule(source: str, *, suppress_benchmark: bool) -> dict[str, object]:
    rule: dict[str, object] = {
        "id": source.replace("/", "-").replace(".", "-"),
        "source": source,
        "disposition": "copy",
        "public_destination": "{source}",
        "reason": "Synthetic copy matching the public Python manifest.",
        "transformation": None,
        "owner": "tests",
        "max_size_bytes": 4096,
        "generated": False,
        "precedence": 100,
    }
    if suppress_benchmark:
        rule["content_classification"] = "trusted-source-code"
        rule["detector_suppressions"] = [RULE_RAW_BENCHMARK_KEY]
    return rule


def _commit(
    tmp_path: Path,
    files: dict[str, str],
    *,
    suppress_benchmark_for: set[str],
) -> tuple[Path, Path, str]:
    repo = tmp_path / "private-source"
    repo.mkdir()
    for relative, text in files.items():
        _write(repo, relative, text)
    manifest = repo / "public-files.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "rules": [
                    _rule("public-files.yaml", suppress_benchmark=False),
                    *[
                        _rule(relative, suppress_benchmark=relative in suppress_benchmark_for)
                        for relative in files
                    ],
                ],
            },
            sort_keys=False,
        ),
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


class _PassingGitleaks:
    def scan_git(self, source: Path, source_sha: str) -> GitleaksEvidence:
        assert source.is_dir()
        return GitleaksEvidence(
            version="8.30.1",
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
            scope="full-history-through-source-sha",
            source_sha=source_sha,
        )


class _PassingFilesystemGitleaks:
    def scan_directory(self, root: Path) -> FilesystemGitleaksEvidence:
        assert root.is_dir()
        return FilesystemGitleaksEvidence(
            version="8.30.1",
            report_sha256=hashlib.sha256(b"[]").hexdigest(),
        )


class _PassingProjectChecks:
    def run(self, root: Path) -> None:
        assert root.is_dir()


def _export_request(
    repo: Path,
    manifest: Path,
    source_sha: str,
    output: Path,
) -> ExportRequest:
    return ExportRequest(
        source=repo,
        output=output,
        manifest=manifest,
        source_sha=source_sha,
        gitleaks_runner=_PassingGitleaks(),
        fleet_hostnames=FLEET_HOSTNAMES,
    )


def _assert_export_refuses(request: ExportRequest, relative: str, rules: frozenset[str]) -> None:
    output = request.output
    try:
        export_public(request)
    except ExportError as error:
        message = str(error)
    else:
        published = (output / relative).read_text(encoding="utf-8")
        pytest.fail(f"export published {relative} without a detector error:\n{published}")

    assert f"detector findings for {relative}" in message
    missing = {rule for rule in rules if rule not in message}
    assert not missing, message
    assert not output.exists()
    leftovers = list(output.parent.glob(f".{output.name}.public-export*"))
    assert leftovers == []


def _published_tree(
    request: ExportRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the tree a scanner bypass would publish, then stop bypassing it."""

    def _hide_findings(*args: object, **kwargs: object) -> list[object]:
        return []

    with monkeypatch.context() as patch:
        patch.setattr(exporter_module, "scan_file", _hide_findings)
        export_public(request)


def _assert_verify_refuses(
    request: ExportRequest,
    relative: str,
    rules: frozenset[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _published_tree(request, monkeypatch)
    published = request.output / relative
    before = published.read_bytes()
    verify_request = PublicVerifyRequest(
        root=request.output,
        source_sha=request.source_sha,
        source_repo=request.source,
        manifest=Path("public-files.yaml"),
        wheelhouse=request.output.parent / "unused-wheelhouse",
        gitleaks_runner=_PassingFilesystemGitleaks(),
        source_gitleaks_runner=_PassingGitleaks(),
        project_checks_runner=_PassingProjectChecks(),
        package_smoke=False,
        fleet_hostnames=FLEET_HOSTNAMES,
    )
    with pytest.raises(PublicVerifyError) as caught:
        verify_public_export(verify_request)
    message = str(caught.value)
    assert published.read_bytes() == before
    assert f"forbidden metadata in {relative}" in message
    missing = {rule for rule in rules if rule not in message}
    assert not missing, message


def test_scan_reports_contextual_rules_inside_src_test_functions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, SRC_RELATIVE, SRC_SOURCE)

    assert _rule_ids(root, SRC_RELATIVE) >= SRC_SCAN_RULES


def test_scan_reports_pyi_and_src_tests_directory_test_functions(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, PYI_RELATIVE, PYI_SOURCE)
    _write(root, SRC_TESTS_RELATIVE, SRC_TESTS_SOURCE)

    assert RULE_PRIVATE_IPV4 in _rule_ids(root, PYI_RELATIVE)
    assert RULE_PRIVATE_IPV4 in _rule_ids(root, SRC_TESTS_RELATIVE)


def test_scan_reports_comment_outside_constant_span_under_tests(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, TESTS_COMMENT_RELATIVE, TESTS_COMMENT_SOURCE)

    assert _rule_ids(root, TESTS_COMMENT_RELATIVE) >= TESTS_COMMENT_RULES


def test_scan_keeps_tests_directory_constant_spans_exempt(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(root, TESTS_LITERAL_RELATIVE, TESTS_LITERAL_SOURCE)

    assert _rule_ids(root, TESTS_LITERAL_RELATIVE).isdisjoint(TESTS_COMMENT_RULES)


def test_scan_still_reports_non_test_and_module_level_src_values(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write(
        root,
        "src/model_forge/runtime.py",
        'def publish() -> None:\n    endpoint = "10.1.2.3"\n',
    )
    _write(
        root,
        "src/model_forge/settings.py",
        'ENDPOINT = "10.1.2.3"\n\ndef test_unrelated() -> None:\n    marker = "safe"\n',
    )

    assert RULE_PRIVATE_IPV4 in _rule_ids(root, "src/model_forge/runtime.py")
    assert RULE_PRIVATE_IPV4 in _rule_ids(root, "src/model_forge/settings.py")


def test_export_refuses_src_test_function_without_publishing(tmp_path: Path) -> None:
    repo, manifest, sha = _commit(
        tmp_path,
        {SRC_RELATIVE: SRC_SOURCE},
        suppress_benchmark_for={SRC_RELATIVE},
    )

    _assert_export_refuses(
        _export_request(repo, manifest, sha, tmp_path / "public"),
        SRC_RELATIVE,
        SRC_PUBLISH_RULES,
    )


def test_export_refuses_tests_comment_outside_constant_span(tmp_path: Path) -> None:
    repo, manifest, sha = _commit(
        tmp_path,
        {TESTS_COMMENT_RELATIVE: TESTS_COMMENT_SOURCE},
        suppress_benchmark_for=set(),
    )

    _assert_export_refuses(
        _export_request(repo, manifest, sha, tmp_path / "public"),
        TESTS_COMMENT_RELATIVE,
        TESTS_COMMENT_RULES,
    )


def test_verify_refuses_src_test_function_leaks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, manifest, sha = _commit(
        tmp_path,
        {SRC_RELATIVE: SRC_SOURCE},
        suppress_benchmark_for={SRC_RELATIVE},
    )

    _assert_verify_refuses(
        _export_request(repo, manifest, sha, tmp_path / "public"),
        SRC_RELATIVE,
        SRC_PUBLISH_RULES,
        monkeypatch,
    )


def test_verify_refuses_tests_comment_outside_constant_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, manifest, sha = _commit(
        tmp_path,
        {TESTS_COMMENT_RELATIVE: TESTS_COMMENT_SOURCE},
        suppress_benchmark_for=set(),
    )

    _assert_verify_refuses(
        _export_request(repo, manifest, sha, tmp_path / "public"),
        TESTS_COMMENT_RELATIVE,
        TESTS_COMMENT_RULES,
        monkeypatch,
    )
