from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from model_forge.public_export.detectors import (
    _KNOWN_RULE_IDS,
    AllowlistEntry,
    DetectorError,
    DetectorPolicy,
    Finding,
    GitleaksStatus,
    ScanWorkBudget,
    apply_allowlist,
    load_fleet_hostname_denylist,
    scan_file,
)

FIXTURES = Path(__file__).with_name("fixtures")
CASES = json.loads((FIXTURES / "detector_cases.json").read_text(encoding="utf-8"))
POLICY = DetectorPolicy(
    max_file_bytes=64,
    max_scan_bytes=64,
    public_contacts=frozenset({"security@example.com"}),
)
SCAN_ERASING_RULES = (
    "credential.gitleaks-required",
    "path.escape",
    "path.forbidden-directory",
    "scan.work-limit",
)


def _text(case: str) -> bytes:
    value = CASES[case]
    if isinstance(value, list):
        value = "".join(str(part) for part in value)
    assert isinstance(value, str)
    return value.encode()


def _scan_text(
    tmp_path: Path,
    case: str,
    *,
    path: str = "docs/example.txt",
    fleet_hostnames: frozenset[str] = frozenset(),
) -> list[Finding]:
    root = tmp_path / "root"
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_text(case))
    return scan_file(
        root,
        path,
        policy=POLICY,
        fleet_hostnames=fleet_hostnames,
        gitleaks_status=GitleaksStatus.PASSED,
    )


@pytest.mark.parametrize(
    ("case", "rule_id"),
    [
        ("rfc1918", "network.private-ipv4"),
        ("cgnat", "network.cgnat-ipv4"),
        ("posix_path", "path.absolute-posix"),
        ("posix_single_component", "path.absolute-posix"),
        ("windows_drive_path", "path.absolute-windows"),
        ("windows_unc_path", "path.absolute-windows"),
        ("nonpublic_email", "identity.nonpublic-email"),
        ("raw_question_key", "benchmark.raw-key"),
        ("raw_response_key", "benchmark.raw-key"),
        ("slack_metadata", "metadata.collaboration"),
        ("discord_metadata", "metadata.collaboration"),
        ("session_metadata", "metadata.session"),
    ],
)
def test_content_detectors_find_positive_fixtures_without_echoing_values(
    tmp_path: Path, case: str, rule_id: str
) -> None:
    findings = _scan_text(tmp_path, case)
    matches = [finding for finding in findings if finding.rule_id == rule_id]

    assert matches
    assert matches[0].path == "docs/example.txt"
    assert matches[0].line is not None
    assert matches[0].offset is not None
    assert _text(case).decode()[-8:] not in matches[0].message


@pytest.mark.parametrize(
    "case",
    [
        "public_network_docs",
        "versions_and_sha",
        "public_urls",
        "public_example_email",
        "allowed_contact",
        "aggregate_evidence_keys",
        "ordinary_metadata",
    ],
)
def test_content_detectors_avoid_bounded_public_false_positives(tmp_path: Path, case: str) -> None:
    assert _scan_text(tmp_path, case) == []


@pytest.mark.parametrize(
    "email",
    [
        "Alice.Smith+release@Corp-Example.com",
        "dev-team@engineering.corp.example",
        "first.last@personal.example",
    ],
)
def test_conventional_dotted_domain_emails_are_detected_case_insensitively(
    tmp_path: Path, email: str
) -> None:
    root = tmp_path / "root"
    target = root / "contacts.txt"
    root.mkdir()
    target.write_text(f"contact={email}\n", encoding="utf-8")

    findings = scan_file(
        root,
        "contacts.txt",
        policy=DetectorPolicy(max_file_bytes=128, max_scan_bytes=128),
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "identity.nonpublic-email" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "reference",
    [
        "owner/repo@revision",
        "actions/checkout@v4",
        "Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "mlabonne/harmful_behaviors@01cead01398926d81f7c52bdb790ee8cf77ebba7",
        "package-name@1.2.3",
        "artifact@0123456789abcdef0123456789abcdef01234567",
        "not/an@email.example",
        ".owner@corp.example",
        "owner..name@corp.example",
        "owner@localhost",
        "owner@invalid_domain.example",
    ],
)
def test_revision_and_package_reference_syntax_is_not_an_email(
    tmp_path: Path, reference: str
) -> None:
    root = tmp_path / "root"
    target = root / "references.txt"
    root.mkdir()
    target.write_text(f"reference={reference}\n", encoding="utf-8")

    findings = scan_file(
        root,
        "references.txt",
        policy=DetectorPolicy(max_file_bytes=128, max_scan_bytes=128),
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "identity.nonpublic-email" not in {finding.rule_id for finding in findings}


def test_localhost_email_detection_requires_explicit_policy(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "contacts.txt"
    root.mkdir()
    target.write_text("contact=operator@localhost\n", encoding="utf-8")

    default_findings = scan_file(
        root,
        "contacts.txt",
        policy=DetectorPolicy(max_file_bytes=64, max_scan_bytes=64),
        gitleaks_status=GitleaksStatus.PASSED,
    )
    localhost_findings = scan_file(
        root,
        "contacts.txt",
        policy=DetectorPolicy(
            max_file_bytes=64,
            max_scan_bytes=64,
            detect_localhost_emails=True,
        ),
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "identity.nonpublic-email" not in {
        finding.rule_id for finding in default_findings
    }
    assert "identity.nonpublic-email" in {
        finding.rule_id for finding in localhost_findings
    }


@pytest.mark.parametrize(
    "text",
    [
        "[Setup](/docs/setup.md)\n",
        '{"pointer": "/components/schemas/Model"}\n',
        r"pattern = r'model\\.language'" + "\n",
        r'pattern = re.compile(r"Answer\s*:\s*([ABCD])\b")' + "\n",
        r'_BOXED_RE = re.compile(r"\\boxed\{\s*([ABCD])\s*\}")' + "\n",
        "endpoint: /api/v1/models\n",
    ],
)
def test_root_relative_public_syntax_is_not_an_absolute_path(tmp_path: Path, text: str) -> None:
    root = tmp_path / "root"
    target = root / "docs/example.txt"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")

    findings = scan_file(
        root,
        "docs/example.txt",
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert not {
        "path.absolute-posix",
        "path.absolute-windows",
    } & {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "text",
    [
        "artifact=/Users/operator/models/release\n",
        "artifact=/home/operator/models/release\n",
        "artifact=/Volumes/Private/models/release\n",
        '{"model_path": "/mnt/private/model"}\n',
        "artifact=C:\\Users\\operator\\models\\release\n",
        "artifact=\\\\build-nas\\private-share\\release.bin\n",
    ],
)
def test_private_operator_absolute_paths_are_detected(tmp_path: Path, text: str) -> None:
    root = tmp_path / "root"
    target = root / "docs/example.txt"
    target.parent.mkdir(parents=True)
    target.write_text(text, encoding="utf-8")

    findings = scan_file(
        root,
        "docs/example.txt",
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert any(finding.rule_id.startswith("path.absolute-") for finding in findings)


def test_python_test_fixture_literals_are_parsed_as_code_context(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "tests/test_private_ip_detector.py"
    target.parent.mkdir(parents=True)
    target.write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('fixture', ['10.1.2.3'])\n"
        "def test_private_ip_detector(fixture: str) -> None:\n"
        "    assert fixture\n",
        encoding="utf-8",
    )

    findings = scan_file(
        root,
        "tests/test_private_ip_detector.py",
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "network.private-ipv4" not in {finding.rule_id for finding in findings}


def test_python_runtime_private_values_remain_blocked(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "src/settings.py"
    target.parent.mkdir(parents=True)
    target.write_text("PRIVATE_ENDPOINT = '10.1.2.3'\n", encoding="utf-8")

    findings = scan_file(
        root,
        "src/settings.py",
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "network.private-ipv4" in {finding.rule_id for finding in findings}


def test_representative_tracked_public_files_have_no_absolute_path_findings() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = [
        "README.md",
        "pyproject.toml",
        "recipes/README.md",
        "contracts/darkstar-release/v1/ledger.schema.json",
    ]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    budget = ScanWorkBudget(max_work_units=4_000_000)

    for path in paths:
        assert path in tracked
        findings = scan_file(
            root,
            path,
            gitleaks_status=GitleaksStatus.PASSED,
            work_budget=budget,
        )
        assert not {
            "path.absolute-posix",
            "path.absolute-windows",
        } & {finding.rule_id for finding in findings}
    assert 0 < budget.used <= budget.max_work_units


def test_tracked_public_revision_syntax_has_no_nonpublic_email_findings() -> None:
    root = Path(__file__).resolve().parents[2]
    paths = [
        ".github/workflows/ci.yml",
        "README.md",
        "configs/modelopt/recipes/nvfp4_mlp_only_mse-kv_bf16.yaml",
        "models/qwen3.8-27b-r3/artifact-lineage.md",
        "models/qwen3.8-27b-r3/model-card/bf16.md",
        "models/qwen3.8-27b-r3/results/publication-readiness-ledger.json",
    ]
    tracked = set(
        subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    budget = ScanWorkBudget(max_work_units=4_000_000)

    for path in paths:
        assert path in tracked
        findings = scan_file(
            root,
            path,
            gitleaks_status=GitleaksStatus.PASSED,
            work_budget=budget,
        )
        assert "identity.nonpublic-email" not in {
            finding.rule_id for finding in findings
        }
    assert 0 < budget.used <= budget.max_work_units


def test_fleet_hostname_denylist_is_loaded_from_bounded_runtime_file(
    tmp_path: Path,
) -> None:
    hostnames = load_fleet_hostname_denylist(FIXTURES / "fleet-hostnames.txt")
    findings = _scan_text(
        tmp_path,
        "fleet_hostname",
        fleet_hostnames=hostnames,
    )

    assert hostnames == frozenset({"forge-worker-07.private", "nas-build.internal"})
    assert [finding.rule_id for finding in findings] == ["identity.fleet-hostname"]
    assert "forge-worker-07.private" not in findings[0].message


@pytest.mark.parametrize(
    "contents",
    [
        b"".join(b"x\n" for _ in range(65_537)),
        b"a" * 1025,
    ],
)
def test_fleet_hostname_file_limits_fail_before_unbounded_work(
    tmp_path: Path, contents: bytes
) -> None:
    denylist = tmp_path / "fleet-hostnames.txt"
    denylist.write_bytes(contents)

    with pytest.raises(DetectorError, match="fleet hostname denylist limit exceeded"):
        load_fleet_hostname_denylist(denylist)


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".envrc",
        ".dockerconfigjson",
        ".npmrc",
        ".pypirc",
        ".netrc",
        "config/secret.yaml",
        "config/secret.yml",
        "config/secret.json",
        "config/secret.toml",
        "config/secrets.toml",
        "config/secret",
        "config/secrets",
        "config/secret.backup",
        "config/secrets.old",
        "config/secret.example",
        "config/secrets.template",
        "config/credentials.sample",
        "config/credentials.json",
        "config/credentials.yaml",
        "config/aws_credentials",
        "config/kubeconfig",
        "config/service-account.json",
        "config/service_account.yaml",
        "config/client_secret.json",
        "config/client-secret.toml",
        "config/application_default_credentials.json",
        "credentials/token.json",
        "credentials/tokens.yml",
        "keys/id_rsa.backup",
        "keys/id_rsa.old",
        "keys/id_ed25519.bak",
        "keys/id_rsa",
        "keys/id_ecdsa_sk",
        "keys/id_ed25519_sk",
        "keys/id_xmss",
        "keys/id_dilithium",
        "keys/id_falcon",
        "keys/id_ml_dsa",
        "keys/id_sphincs",
        "keys/id_sphincsplus",
        "keys/deploy-private-key",
        "certs/private.pem",
        "certs/release.JKS",
        "certs/client.keystore",
        "auth/service.keytab",
        "infra/terraform.tfvars",
        "infra/production.auto.tfvars",
        "credentials/token.txt",
        "credentials/auth.json",
        "CONFIG/.ENV.STAGING",
    ],
)
def test_secret_like_filename_detector(path: str, tmp_path: Path) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=path)
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "path",
    [
        "docs/secret.example.yaml",
        "docs/secrets.template.json",
        "examples/auth.sample.json",
        "examples/token.example.txt",
        "infra/terraform.example.tfvars",
        "infra/example.auto.tfvars",
        "infra/production.template.auto.tfvars",
        "records/id.txt",
        "schemas/id.json",
        "keys/id_public",
        "keys/id_model",
        "keys/id_rsa.pub",
        "keys/id_ed25519.pub",
        "keys/id_ecdsa_sk.pub",
        "keys/id_ed25519.pub.backup",
        "docs/credentials.example.yaml",
        "docs/service_account.template.json",
        "docs/client_secret.sample.toml",
        "docs/application_default_credentials.example.json",
        "docs/tokens.template.yml",
        "docs/changelog.old",
        "notes/meeting.bak",
    ],
)
def test_explicit_public_templates_and_ordinary_id_files_are_not_secret_like(
    path: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=path)
    assert "filename.secret-like" not in {finding.rule_id for finding in findings}


SECRET_CONFIG_STEMS = (
    "application_default_credentials",
    "application-default-credentials",
    "client-secret",
    "client_secret",
    "credentials",
    "secret",
    "secrets",
    "service-account",
    "service_account",
    "token",
    "tokens",
)
SECRET_CONFIG_EXTENSIONS = ("json", "toml", "yaml", "yml")
PUBLIC_TEMPLATE_MARKERS = ("example", "sample", "template")
BACKUP_SUFFIXES = (
    "~",
    ".backup",
    ".bak",
    ".copy",
    ".old",
    ".orig",
    ".save",
    ".swp",
    ".tmp",
)
PRIVATE_KEY_BASENAMES = (
    "id_dilithium",
    "id_dsa",
    "id_ecdsa",
    "id_ecdsa_sk",
    "id_ed25519",
    "id_ed25519_sk",
    "id_falcon",
    "id_ml_dsa",
    "id_rsa",
    "id_sphincs",
    "id_sphincsplus",
    "id_xmss",
)
STACKED_BACKUP_SUFFIX_CHAINS = (
    "~.orig",
    ".bak~",
    ".old.backup",
    ".swp.tmp",
    ".copy.save",
)


@pytest.mark.parametrize("extension", SECRET_CONFIG_EXTENSIONS)
@pytest.mark.parametrize("stem", SECRET_CONFIG_STEMS)
def test_secret_credential_basename_matrix_is_secret_like(
    stem: str, extension: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"config/{stem.upper()}.{extension}")
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("suffix", BACKUP_SUFFIXES)
@pytest.mark.parametrize("basename", PRIVATE_KEY_BASENAMES)
def test_private_key_backup_variant_matrix_is_secret_like(
    basename: str, suffix: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"keys/{basename}{suffix}")
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("suffix", ("",) + BACKUP_SUFFIXES + STACKED_BACKUP_SUFFIX_CHAINS)
@pytest.mark.parametrize("basename", PRIVATE_KEY_BASENAMES)
def test_public_key_matrix_is_not_secret_like(basename: str, suffix: str, tmp_path: Path) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"keys/{basename}.pub{suffix}")
    assert "filename.secret-like" not in {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "path",
    [
        "keys/ID_RSA.PUB~",
        "keys/ID_ED25519.PUB~",
        "keys/backups/id_rsa.pub~.orig",
        "keys/backups/id_ed25519.pub.bak",
        "keys/nested/deep/id_ecdsa_sk.Pub.Bak",
        "Keys/Backups/ID_ED25519.PUB.BAK~",
    ],
)
def test_public_key_backup_case_and_nested_variants_are_not_secret_like(
    path: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=path)
    assert "filename.secret-like" not in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("suffix", STACKED_BACKUP_SUFFIX_CHAINS)
@pytest.mark.parametrize("basename", PRIVATE_KEY_BASENAMES)
def test_stacked_private_key_backup_chain_matrix_is_secret_like(
    basename: str, suffix: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"keys/{basename}{suffix}")
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "path",
    [
        "keys/ID_RSA~",
        "keys/backups/id_rsa.old",
        "keys/backups/ID_ED25519.BAK",
        "keys/nested/deep/id_xmss.Backup",
        "keys/id_rsa.pub.1",
        "keys/id_rsa.pub.enc",
        "keys/id_ed25519.pub.tar.gz",
        "keys/id_rsa.pub.bak.old.copy",
        "keys/id_ed25519.pub~.orig.bak",
    ],
)
def test_private_key_backups_and_unrecognized_public_key_suffixes_fail_closed(
    path: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=path)
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("suffix", BACKUP_SUFFIXES)
@pytest.mark.parametrize(
    "path",
    [
        ".env",
        "config/secret",
        "config/secrets",
        "config/secrets.json",
        "config/credentials.yaml",
        "config/service_account.json",
        "credentials/token.json",
        "infra/terraform.tfvars",
        "certs/private.pem",
    ],
)
def test_secret_backup_variant_matrix_is_secret_like(
    path: str, suffix: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"{path}{suffix}")
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("suffix", BACKUP_SUFFIXES)
@pytest.mark.parametrize(
    "path",
    [
        "docs/secret.example.yaml",
        "docs/secrets.template.json",
        "docs/credentials.sample.toml",
        "examples/auth.sample.json",
        "examples/token.example.txt",
        "infra/terraform.example.tfvars",
        "infra/example.auto.tfvars",
        "infra/production.template.auto.tfvars",
    ],
)
def test_template_named_secret_backups_cannot_evade_detection(
    path: str, suffix: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"{path}{suffix}")
    assert "filename.secret-like" in {finding.rule_id for finding in findings}


@pytest.mark.parametrize("extension", SECRET_CONFIG_EXTENSIONS)
@pytest.mark.parametrize("marker", PUBLIC_TEMPLATE_MARKERS)
@pytest.mark.parametrize("stem", SECRET_CONFIG_STEMS)
def test_documented_public_template_matrix_is_not_secret_like(
    stem: str, marker: str, extension: str, tmp_path: Path
) -> None:
    findings = _scan_text(tmp_path, "ordinary_metadata", path=f"docs/{stem}.{marker}.{extension}")
    assert "filename.secret-like" not in {finding.rule_id for finding in findings}


@pytest.mark.parametrize(
    "path",
    [
        ".hermes/plan.md",
        ".git/config",
        "models/demo/results/raw/output.json",
        "src/model_forge/__pycache__/module.pyc",
        ".pytest_cache/state",
    ],
)
def test_forbidden_directory_detector_does_not_need_path_to_exist(
    path: str, tmp_path: Path
) -> None:
    findings = scan_file(
        tmp_path,
        path,
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )
    assert [finding.rule_id for finding in findings] == ["path.forbidden-directory"]


@pytest.mark.parametrize(
    "path",
    ["../outside.txt", "/absolute.txt", "docs/../outside.txt", r"docs\outside.txt"],
)
def test_noncanonical_or_escaping_paths_fail_closed(path: str, tmp_path: Path) -> None:
    findings = scan_file(
        tmp_path,
        path,
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )
    assert [finding.rule_id for finding in findings] == ["path.escape"]


def test_binary_and_oversized_files_are_rejected_before_content_scanning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(b"public\x00binary")
    (root / "large.txt").write_bytes(b"x" * 65)

    binary = scan_file(root, "binary.dat", policy=POLICY, gitleaks_status=GitleaksStatus.PASSED)
    oversized = scan_file(root, "large.txt", policy=POLICY, gitleaks_status=GitleaksStatus.PASSED)

    assert [finding.rule_id for finding in binary] == ["file.binary"]
    assert [finding.rule_id for finding in oversized] == ["file.oversized"]


def test_known_binary_magic_is_rejected_even_without_nul_bytes(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "document.pdf").write_bytes(b"%PDF-1.7\nsynthetic fixture")

    findings = scan_file(root, "document.pdf", policy=POLICY, gitleaks_status=GitleaksStatus.PASSED)

    assert [finding.rule_id for finding in findings] == ["file.binary"]


@pytest.mark.parametrize(
    "contents",
    [
        b"GIF87a synthetic",
        b"GIF89a synthetic",
        b"RIFF\x08\x00\x00\x00WEBPsynthetic",
        b"UTF-8 decodable\x01control",
    ],
)
def test_additional_binary_signatures_and_control_bytes_are_rejected(
    tmp_path: Path, contents: bytes
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "binary.dat").write_bytes(contents)

    findings = scan_file(root, "binary.dat", policy=POLICY, gitleaks_status=GitleaksStatus.PASSED)

    assert [finding.rule_id for finding in findings] == ["file.binary"]


@pytest.mark.parametrize(
    ("name", "contents"),
    [
        ("binary.dat", b"public\x00binary"),
        ("large.txt", b"x" * 65),
    ],
)
def test_gitleaks_not_run_is_reported_for_unscannable_files(
    tmp_path: Path, name: str, contents: bytes
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / name).write_bytes(contents)

    findings = scan_file(root, name, policy=POLICY)

    assert "credential.gitleaks-required" in {finding.rule_id for finding in findings}


def test_gitleaks_requirement_is_structured_and_never_faked(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_text("ordinary public text", encoding="utf-8")

    findings = scan_file(root, "safe.txt", policy=POLICY)

    assert [finding.rule_id for finding in findings] == ["credential.gitleaks-required"]
    assert findings[0].severity == "error"
    assert findings[0].line is None
    assert findings[0].offset is None


def test_unknown_gitleaks_status_cannot_suppress_requirement(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_text("ordinary public text", encoding="utf-8")

    with pytest.raises(DetectorError, match="Gitleaks status"):
        scan_file(
            root,
            "safe.txt",
            policy=POLICY,
            gitleaks_status="passed",  # type: ignore[arg-type]
        )


def test_findings_are_stably_ordered_and_structured(tmp_path: Path) -> None:
    findings = _scan_text(tmp_path, "multiple_findings")
    assert findings == sorted(
        findings,
        key=lambda finding: (
            finding.path,
            finding.offset if finding.offset is not None else -1,
            finding.rule_id,
        ),
    )
    assert all(finding.severity in {"error", "warning"} for finding in findings)


def test_multibyte_locations_are_byte_offsets_and_line_numbers(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "locations.txt"
    root.mkdir()
    target.write_text("é\npeer=10.1.2.3 owner=person@internal.invalid\n", encoding="utf-8")

    findings = scan_file(
        root,
        "locations.txt",
        policy=DetectorPolicy(max_file_bytes=128, max_scan_bytes=128),
        gitleaks_status=GitleaksStatus.PASSED,
    )

    located = {finding.rule_id: (finding.line, finding.offset) for finding in findings}
    assert located["network.private-ipv4"] == (2, 8)
    assert located["identity.nonpublic-email"] == (2, 23)


def test_repeated_ip_input_has_deterministic_bounded_work_and_findings(
    tmp_path: Path,
) -> None:
    size = 1_048_576
    contents = (b"10.0.0.1 " * ((size // 9) + 1))[:size]
    root = tmp_path / "root"
    root.mkdir()
    (root / "adversarial.txt").write_bytes(contents)
    policy = DetectorPolicy(
        max_file_bytes=size,
        max_scan_bytes=size,
        max_findings_per_file=32,
        max_work_units_per_file=4_000_000,
    )
    budget = ScanWorkBudget(max_work_units=4_000_000)

    findings = scan_file(
        root,
        "adversarial.txt",
        policy=policy,
        gitleaks_status=GitleaksStatus.PASSED,
        work_budget=budget,
    )

    assert [finding.rule_id for finding in findings].count("network.private-ipv4") == 32
    assert [finding.rule_id for finding in findings].count("scan.work-limit") == 1
    assert len(findings) == 33
    assert budget.used == (3 * size) + 33


def test_repository_work_budget_is_shared_and_never_exceeded(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for name in ("one.txt", "two.txt"):
        (root / name).write_text("ordinary public text", encoding="utf-8")
    budget = ScanWorkBudget(max_work_units=240)

    first = scan_file(
        root,
        "one.txt",
        gitleaks_status=GitleaksStatus.PASSED,
        work_budget=budget,
    )
    second = scan_file(
        root,
        "two.txt",
        gitleaks_status=GitleaksStatus.PASSED,
        work_budget=budget,
    )

    assert [finding.rule_id for finding in first] == []
    assert [finding.rule_id for finding in second] == ["scan.work-limit"]
    assert budget.used <= budget.max_work_units
    assert second[0].line is None
    assert second[0].offset is None
    assert "ordinary public text" not in second[0].message


def test_allowlist_is_exact_path_and_rule_scoped(tmp_path: Path) -> None:
    finding = _scan_text(tmp_path, "rfc1918")[0]
    entry = AllowlistEntry(
        path="docs/example.txt",
        rule_id="network.private-ipv4",
        justification="Public documentation of an intentionally private range.",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )

    assert apply_allowlist([finding], [entry]) == []
    assert apply_allowlist(
        [finding],
        [
            AllowlistEntry(
                path="docs/other.txt",
                rule_id=entry.rule_id,
                justification=entry.justification,
                expires_at=entry.expires_at,
            )
        ],
    ) == [finding]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"path": "*"}, "wildcards"),
        ({"path": "**/*"}, "wildcards"),
        ({"path": "../docs/example.txt"}, "canonical"),
        ({"path": "docs/\x00example.txt"}, "NUL"),
        ({"path": "docs/" + ("x" * 4096)}, "length"),
        ({"rule_id": "*"}, "wildcards"),
        ({"rule_id": "network.typo"}, "known"),
        ({"justification": "  "}, "justification"),
        ({"expires_at": datetime.now()}, "timezone-aware"),
        ({"expires_at": datetime.now(UTC) - timedelta(seconds=1)}, "expired"),
        ({"rule_id": "credential.gitleaks-required"}, "cannot be allowlisted"),
        ({"rule_id": "path.escape"}, "cannot be allowlisted"),
        ({"rule_id": "path.forbidden-directory"}, "cannot be allowlisted"),
        ({"rule_id": "scan.work-limit"}, "cannot be allowlisted"),
    ],
)
def test_invalid_or_dangerous_allowlist_entries_fail_closed(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "path": "docs/example.txt",
        "rule_id": "network.private-ipv4",
        "justification": "Narrow temporary exception.",
        "expires_at": datetime.now(UTC) + timedelta(days=1),
    }
    values.update(overrides)

    with pytest.raises(DetectorError, match=message):
        AllowlistEntry(**values)  # type: ignore[arg-type]


def _entry_skipping_construction_checks(path: str, rule_id: str) -> AllowlistEntry:
    entry = object.__new__(AllowlistEntry)
    values: dict[str, object] = {
        "path": path,
        "rule_id": rule_id,
        "justification": "Reviewed and approved by the export owner.",
        "expires_at": datetime.now(UTC) + timedelta(days=1),
    }
    for name, value in values.items():
        object.__setattr__(entry, name, value)
    return entry


@pytest.mark.parametrize("rule_id", SCAN_ERASING_RULES)
def test_rules_that_stop_content_scanning_are_never_allowlistable(rule_id: str) -> None:
    with pytest.raises(DetectorError, match="cannot be allowlisted"):
        AllowlistEntry(
            path="docs/example.txt",
            rule_id=rule_id,
            justification="Reviewed and approved by the export owner.",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
    finding = Finding(rule_id=rule_id, path="docs/example.txt", message="redacted")
    smuggled = _entry_skipping_construction_checks("docs/example.txt", rule_id)
    with pytest.raises(DetectorError, match="cannot be allowlisted"):
        apply_allowlist([finding], [smuggled])


def test_every_known_rule_is_hard_blocked_or_reviewable() -> None:
    assert set(SCAN_ERASING_RULES) <= _KNOWN_RULE_IDS
    for rule_id in sorted(_KNOWN_RULE_IDS - set(SCAN_ERASING_RULES)):
        finding = Finding(rule_id=rule_id, path="docs/example.txt", message="redacted")
        entry = AllowlistEntry(
            path="docs/example.txt",
            rule_id=rule_id,
            justification="Reviewed and approved by the export owner.",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
        assert apply_allowlist([finding], [entry]) == []


@pytest.mark.parametrize("path", [".git/config", "models/demo/results/raw/output.json"])
def test_forbidden_directory_finding_cannot_be_suppressed(path: str, tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("peer=10.1.2.3 owner=person@internal.invalid\n", encoding="utf-8")

    findings = scan_file(root, path, policy=POLICY, gitleaks_status=GitleaksStatus.PASSED)

    assert [finding.rule_id for finding in findings] == ["path.forbidden-directory"]
    with pytest.raises(DetectorError, match="cannot be allowlisted"):
        AllowlistEntry(
            path=path,
            rule_id="path.forbidden-directory",
            justification="Needed for the public export of historical evidence.",
            expires_at=datetime.now(UTC) + timedelta(days=1),
        )
    unrelated = AllowlistEntry(
        path=path,
        rule_id="network.private-ipv4",
        justification="Documented private range in reviewed content.",
        expires_at=datetime.now(UTC) + timedelta(days=1),
    )
    assert apply_allowlist(findings, [unrelated]) == findings


def test_policy_and_scan_inputs_have_hard_bounds(tmp_path: Path) -> None:
    with pytest.raises(DetectorError, match="max_file_bytes"):
        DetectorPolicy(max_file_bytes=0)
    with pytest.raises(DetectorError, match="public contacts limit"):
        DetectorPolicy(
            public_contacts=frozenset(f"user{index}@example.test" for index in range(257))
        )

    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_text("safe", encoding="utf-8")
    with pytest.raises(DetectorError, match="fleet hostname limit"):
        scan_file(
            root,
            "safe.txt",
            policy=POLICY,
            fleet_hostnames=frozenset(f"host-{index}.internal" for index in range(257)),
            gitleaks_status=GitleaksStatus.PASSED,
        )


@pytest.mark.parametrize(
    "text",
    [
        "sessionId: private\n",
        "session id: private\n",
        "session.id: private\n",
        "conversation-id=private\n",
        "chatSessionId: private\n",
        "cursor-session-id=private\n",
        '{"sessionId": "private"}\n',
    ],
)
def test_session_metadata_key_variants_are_detected(tmp_path: Path, text: str) -> None:
    root = tmp_path / "root"
    target = root / "metadata.txt"
    root.mkdir()
    target.write_text(text, encoding="utf-8")

    findings = scan_file(
        root,
        "metadata.txt",
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "metadata.session" in {finding.rule_id for finding in findings}


def test_slack_client_url_is_collaboration_metadata(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "metadata.txt"
    root.mkdir()
    target.write_text(
        "https://app.slack.com/client/T123/C456/thread-789\n",
        encoding="utf-8",
    )

    findings = scan_file(
        root,
        "metadata.txt",
        policy=POLICY,
        gitleaks_status=GitleaksStatus.PASSED,
    )

    assert "metadata.collaboration" in {finding.rule_id for finding in findings}
