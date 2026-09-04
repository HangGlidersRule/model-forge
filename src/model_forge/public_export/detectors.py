"""Bounded, fail-closed detectors for public repository exports."""

from __future__ import annotations

import ast
import ipaddress
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal, Pattern

MAX_POLICY_BYTES = 16_777_216
MAX_PUBLIC_CONTACTS = 256
MAX_FLEET_HOSTNAMES = 256
MAX_DENYLIST_BYTES = 1_024
MAX_ALLOWLIST_ENTRIES = 256
MAX_IDENTIFIER_LENGTH = 254
MAX_ALLOWLIST_PATH_LENGTH = 4_096
MAX_FINDINGS_PER_FILE = 4_096
MAX_SCAN_WORK_UNITS = 268_435_456
MAX_BACKUP_SUFFIX_STRIPS = 2

Severity = Literal["error", "warning"]

RULE_PRIVATE_IPV4 = "network.private-ipv4"
RULE_CGNAT_IPV4 = "network.cgnat-ipv4"
RULE_POSIX_PATH = "path.absolute-posix"
RULE_WINDOWS_PATH = "path.absolute-windows"
RULE_PATH_ESCAPE = "path.escape"
RULE_FLEET_HOSTNAME = "identity.fleet-hostname"
RULE_NONPUBLIC_EMAIL = "identity.nonpublic-email"
RULE_SECRET_FILENAME = "filename.secret-like"
RULE_RAW_BENCHMARK_KEY = "benchmark.raw-key"
RULE_COLLABORATION_METADATA = "metadata.collaboration"
RULE_SESSION_METADATA = "metadata.session"
RULE_OVERSIZED_FILE = "file.oversized"
RULE_BINARY_FILE = "file.binary"
RULE_FORBIDDEN_DIRECTORY = "path.forbidden-directory"
RULE_GITLEAKS_REQUIRED = "credential.gitleaks-required"
RULE_SCAN_WORK_LIMIT = "scan.work-limit"

_IPV4_CANDIDATE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_POSIX_PATH = re.compile(
    r"(?<![/:A-Za-z0-9._-])/(?:Users|home)/[^/\s\"'<>]+"
    r"(?:/[^/\s\"'<>]+)*|"
    r"(?<![/:A-Za-z0-9._-])/Volumes/[^/\s\"'<>]+(?:/[^/\s\"'<>]+)*"
)
_STRUCTURED_POSIX_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[\"']?(?:artifact|cache|directory|dir|file_path|"
    r"filepath|model_path|output_path|path|root|workspace)[\"']?"
    r"(?![A-Za-z0-9_])[ \t]*[:=][ \t]*[\"']?"
    r"(?P<path>/(?!/)[^/\s\"'<>]+(?:/[^/\s\"'<>]+)*)"
)
_WINDOWS_DRIVE_PATH = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])"
    r"[^\\/\s\"'<>]+(?:[\\/][^\\/\s\"'<>]+)*"
)
_WINDOWS_UNC_PATH = re.compile(
    r"(?<![A-Za-z0-9_\\])\\\\[A-Za-z0-9][A-Za-z0-9._-]*[\\/]"
    r"[A-Za-z0-9][^\\/\s\"'<>]*(?:[\\/][^\\/\s\"'<>]+)*"
)
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._+/-])"
    r"[A-Za-z0-9]+(?:[._+-][A-Za-z0-9]+)*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?"
    r"(?![A-Za-z0-9.-])"
)
_RAW_KEY = re.compile(
    r"(?im)(?:^[ \t]*(?:[-{,][ \t]*)?|[{,][ \t]*)"
    r"[\"']?(?:question|answer|prompt|response|answer_key|correct_answer|"
    r"model_response|raw_response)[\"']?[ \t]*[:=]"
)
_COLLABORATION_METADATA = re.compile(
    r"(?i)(?:https?://app\.slack\.com/client/|"
    r"https?://(?:[^/\s]+\.)?slack\.com/archives/|"
    r"https?://discord\.com/channels/|"
    r"\b(?:slack_thread_ts|slack_channel_id|discord_channel_id|"
    r"discord_message_id)[\"']?[ \t]*[:=])"
)
_SESSION_METADATA = re.compile(
    r"(?i)\b(?:session[_. -]?id|conversation[_-]?id|"
    r"chat[_-]?session[_-]?id|cursor[_-]?session[_-]?id)"
    r"[\"']?[ \t]*[:=]"
)
_HOSTNAME = re.compile(
    r"(?=.{1,253}\Z)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)*"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
)
_SECRET_BASENAMES = {
    ".env",
    ".env.local",
    ".env.production",
    ".envrc",
    ".dockerconfigjson",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "auth.json",
    "aws_credentials",
    "credentials",
    "credentials.json",
    "kubeconfig",
    "private-key",
    "private_key",
    "privatekey",
    "secret",
    "secrets",
    "service-account.json",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "terraform.tfvars",
    "token.txt",
}
_PRIVATE_KEY_ID_BASENAMES = {
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
}
_SECRET_CREDENTIAL_BASENAME = re.compile(
    r"(?:application[-_]default[-_]credentials"
    r"|client[-_]secret"
    r"|credentials"
    r"|secrets?"
    r"|service[-_]account"
    r"|tokens?"
    r")\.(?:json|toml|ya?ml)"
)
_PUBLIC_TEMPLATE_MARKERS = frozenset({"example", "sample", "template"})
_BACKUP_SUFFIXES = (
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
_PUBLIC_KEY_COMPONENT = "pub"
_SECRET_SUFFIXES = {
    ".env",
    ".jks",
    ".key",
    ".keytab",
    ".keystore",
    ".p12",
    ".pfx",
    ".pem",
}
_FORBIDDEN_COMPONENTS = {
    ".cache",
    ".git",
    ".hermes",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
}
_BINARY_MAGIC = (
    b"%PDF-",
    b"\x1f\x8b",
    b"\x7fELF",
    b"\x89PNG\r\n\x1a\n",
    b"PK\x03\x04",
    b"GIF87a",
    b"GIF89a",
)
_NON_ALLOWLISTABLE_RULES = {
    RULE_GITLEAKS_REQUIRED,
    RULE_PATH_ESCAPE,
    RULE_FORBIDDEN_DIRECTORY,
    RULE_SCAN_WORK_LIMIT,
}
_KNOWN_RULE_IDS = {
    RULE_PRIVATE_IPV4,
    RULE_CGNAT_IPV4,
    RULE_POSIX_PATH,
    RULE_WINDOWS_PATH,
    RULE_PATH_ESCAPE,
    RULE_FLEET_HOSTNAME,
    RULE_NONPUBLIC_EMAIL,
    RULE_SECRET_FILENAME,
    RULE_RAW_BENCHMARK_KEY,
    RULE_COLLABORATION_METADATA,
    RULE_SESSION_METADATA,
    RULE_OVERSIZED_FILE,
    RULE_BINARY_FILE,
    RULE_FORBIDDEN_DIRECTORY,
    RULE_GITLEAKS_REQUIRED,
    RULE_SCAN_WORK_LIMIT,
}
_PYTHON_CONTEXTUAL_RULES = {
    RULE_PRIVATE_IPV4,
    RULE_CGNAT_IPV4,
    RULE_POSIX_PATH,
    RULE_WINDOWS_PATH,
    RULE_FLEET_HOSTNAME,
    RULE_NONPUBLIC_EMAIL,
    RULE_RAW_BENCHMARK_KEY,
    RULE_COLLABORATION_METADATA,
    RULE_SESSION_METADATA,
}


class DetectorError(ValueError):
    """Detector configuration or input violates a bounded security contract."""


class GitleaksStatus(str, Enum):
    """Externally established Gitleaks status for one scan."""

    NOT_RUN = "not-run"
    PASSED = "passed"


@dataclass(frozen=True, slots=True)
class Finding:
    """A redacted detector result with a stable machine-readable identity."""

    rule_id: str
    path: str
    message: str
    severity: Severity = "error"
    line: int | None = None
    offset: int | None = None


@dataclass(slots=True)
class ScanWorkBudget:
    """A deterministic work ceiling shareable across repository file scans."""

    max_work_units: int = 67_108_864
    used: int = 0

    def __post_init__(self) -> None:
        if (
            type(self.max_work_units) is not int
            or self.max_work_units <= 0
            or self.max_work_units > MAX_SCAN_WORK_UNITS
        ):
            raise DetectorError(
                f"max_work_units must be a positive integer no greater than {MAX_SCAN_WORK_UNITS}"
            )
        if type(self.used) is not int or self.used < 0 or self.used > self.max_work_units:
            raise DetectorError("used work must be an integer within max_work_units")


@dataclass(frozen=True, slots=True)
class DetectorPolicy:
    """Fixed resource limits and explicitly public contacts."""

    max_file_bytes: int = 1_048_576
    max_scan_bytes: int = 1_048_576
    max_findings_per_file: int = 256
    max_work_units_per_file: int = 16_777_216
    public_contacts: frozenset[str] = frozenset()
    detect_localhost_emails: bool = False

    def __post_init__(self) -> None:
        _validate_positive_bound("max_file_bytes", self.max_file_bytes)
        _validate_positive_bound("max_scan_bytes", self.max_scan_bytes)
        if (
            type(self.max_findings_per_file) is not int
            or self.max_findings_per_file <= 0
            or self.max_findings_per_file > MAX_FINDINGS_PER_FILE
        ):
            raise DetectorError(
                "max_findings_per_file must be a positive integer no greater than "
                f"{MAX_FINDINGS_PER_FILE}"
            )
        if (
            type(self.max_work_units_per_file) is not int
            or self.max_work_units_per_file <= 0
            or self.max_work_units_per_file > MAX_SCAN_WORK_UNITS
        ):
            raise DetectorError(
                "max_work_units_per_file must be a positive integer no greater than "
                f"{MAX_SCAN_WORK_UNITS}"
            )
        if self.max_scan_bytes > self.max_file_bytes:
            raise DetectorError("max_scan_bytes cannot exceed max_file_bytes")
        if type(self.detect_localhost_emails) is not bool:
            raise DetectorError("detect_localhost_emails must be a boolean")
        if len(self.public_contacts) > MAX_PUBLIC_CONTACTS:
            raise DetectorError(f"public contacts limit exceeded: {MAX_PUBLIC_CONTACTS}")
        normalized: set[str] = set()
        for contact in self.public_contacts:
            if (
                not isinstance(contact, str)
                or len(contact) > MAX_IDENTIFIER_LENGTH
                or _EMAIL.fullmatch(contact) is None
                or not _is_conventional_email(contact, self.detect_localhost_emails)
            ):
                raise DetectorError("public contact must be a bounded email address")
            normalized.add(contact.casefold())
        object.__setattr__(self, "public_contacts", frozenset(normalized))


@dataclass(frozen=True, slots=True)
class AllowlistEntry:
    """An exact, temporary exception for one path and one detector rule."""

    path: str
    rule_id: str
    justification: str
    expires_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.path, str):
            raise DetectorError("allowlist path must be a string")
        if not isinstance(self.rule_id, str):
            raise DetectorError("allowlist rule_id must be a string")
        if "\0" in self.path:
            raise DetectorError("allowlist path must not contain NUL")
        if len(self.path) > MAX_ALLOWLIST_PATH_LENGTH:
            raise DetectorError(
                f"allowlist path length limit exceeded: {MAX_ALLOWLIST_PATH_LENGTH}"
            )
        if "*" in self.path or "?" in self.path or "*" in self.rule_id or "?" in self.rule_id:
            raise DetectorError("allowlist wildcards are forbidden")
        if not _is_canonical_relative_path(self.path):
            raise DetectorError("allowlist path must be canonical and repository-relative")
        if not self.rule_id or len(self.rule_id) > MAX_IDENTIFIER_LENGTH:
            raise DetectorError("allowlist rule_id must be nonempty and bounded")
        if self.rule_id not in _KNOWN_RULE_IDS:
            raise DetectorError("allowlist rule_id must be a known detector rule")
        if not self.justification.strip():
            raise DetectorError("allowlist justification must be nonempty")
        if len(self.justification) > 1_024:
            raise DetectorError("allowlist justification limit exceeded: 1024")
        if not isinstance(self.expires_at, datetime):
            raise DetectorError("allowlist expiration must be a datetime")
        if self.expires_at.tzinfo is None or self.expires_at.utcoffset() is None:
            raise DetectorError("allowlist expiration must be timezone-aware")
        if self.expires_at <= datetime.now(UTC):
            raise DetectorError("allowlist entry is expired")
        _reject_non_allowlistable_rule(self.rule_id)


def _reject_non_allowlistable_rule(rule_id: str) -> None:
    if rule_id in _NON_ALLOWLISTABLE_RULES or rule_id.startswith("credential."):
        raise DetectorError(f"{rule_id} cannot be allowlisted")


def _validate_positive_bound(name: str, value: int) -> None:
    if type(value) is not int or value <= 0 or value > MAX_POLICY_BYTES:
        raise DetectorError(f"{name} must be a positive integer no greater than {MAX_POLICY_BYTES}")


def _is_canonical_relative_path(value: str) -> bool:
    if (
        not value
        or "\0" in value
        or "\\" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value)
    ):
        return False
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        return False
    return PurePosixPath(value).as_posix() == value


def _path_finding(rule_id: str, path: str, message: str) -> Finding:
    return Finding(rule_id=rule_id, path=path, message=message)


def _has_forbidden_directory(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if any(part.casefold() in _FORBIDDEN_COMPONENTS for part in parts):
        return True
    lowered = tuple(part.casefold() for part in parts)
    return any(
        lowered[index : index + 2] == ("results", "raw") for index in range(len(lowered) - 1)
    )


def _is_documented_public_template(basename: str) -> bool:
    public_template_suffixes = (
        ".example.json",
        ".example.tfvars",
        ".example.toml",
        ".example.txt",
        ".example.yaml",
        ".example.yml",
        ".sample.json",
        ".sample.tfvars",
        ".sample.toml",
        ".sample.txt",
        ".sample.yaml",
        ".sample.yml",
        ".template.json",
        ".template.tfvars",
        ".template.toml",
        ".template.txt",
        ".template.yaml",
        ".template.yml",
    )
    auto_tfvars_template = re.fullmatch(
        r"(?:.+\.)?(?:example|sample|template)\.auto\.tfvars", basename
    )
    return auto_tfvars_template is not None or basename.endswith(public_template_suffixes)


def _normalized_key_basename(basename: str) -> str:
    """Peel at most MAX_BACKUP_SUFFIX_STRIPS recognized backup or editor suffixes."""

    normalized = basename
    for _ in range(MAX_BACKUP_SUFFIX_STRIPS):
        peeled = _strip_backup_suffix(normalized)
        if peeled is None:
            break
        normalized = peeled
    return normalized


def _is_private_key_basename(basename: str) -> bool:
    normalized = _normalized_key_basename(basename)
    stem, separator, _ = normalized.partition(".")
    if stem not in _PRIVATE_KEY_ID_BASENAMES:
        return False
    if not separator:
        return True
    return normalized != f"{stem}.{_PUBLIC_KEY_COMPONENT}"


def _without_template_markers(basename: str) -> str:
    components = basename.split(".")
    kept = [component for component in components if component not in _PUBLIC_TEMPLATE_MARKERS]
    stripped = ".".join(kept)
    if not stripped:
        return basename
    return stripped


def _strip_backup_suffix(basename: str) -> str | None:
    for suffix in _BACKUP_SUFFIXES:
        if basename.endswith(suffix) and len(basename) > len(suffix):
            return basename[: -len(suffix)]
    return None


def _secret_name_candidates(basename: str) -> tuple[str, ...]:
    candidates: list[str] = []
    current: str | None = basename
    for _ in range(MAX_BACKUP_SUFFIX_STRIPS + 1):
        if current is None:
            break
        candidates.append(current)
        unmarked = _without_template_markers(current)
        if unmarked != current:
            candidates.append(unmarked)
        current = _strip_backup_suffix(current)
    return tuple(candidates)


def _is_secret_basename(basename: str) -> bool:
    return (
        basename in _SECRET_BASENAMES
        or _is_private_key_basename(basename)
        or any(basename.endswith(suffix) for suffix in _SECRET_SUFFIXES)
        or basename.startswith(".env.")
        or _SECRET_CREDENTIAL_BASENAME.fullmatch(basename) is not None
        or basename.endswith(".auto.tfvars")
        or re.search(r"(?:^|[._-])private[._-]?key(?:[._-]|$)", basename) is not None
    )


def _secret_like_filename(path: str) -> bool:
    basename = PurePosixPath(path).name.casefold()
    if _is_documented_public_template(basename):
        return False
    return any(_is_secret_basename(candidate) for candidate in _secret_name_candidates(basename))


def load_fleet_hostname_denylist(path: Path) -> frozenset[str]:
    """Load a small private hostname file without logging its values."""

    try:
        size = path.stat().st_size
    except OSError as error:
        raise DetectorError("fleet hostname denylist cannot be read") from error
    if size > MAX_DENYLIST_BYTES:
        raise DetectorError(f"fleet hostname denylist limit exceeded: {MAX_DENYLIST_BYTES} bytes")
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise DetectorError("fleet hostname denylist must be readable UTF-8") from error

    hostnames: set[str] = set()
    for line in text.splitlines():
        candidate = line.strip().casefold()
        if not candidate or candidate.startswith("#"):
            continue
        if (
            len(candidate) > MAX_IDENTIFIER_LENGTH
            or _HOSTNAME.fullmatch(candidate) is None
            or "." not in candidate
        ):
            raise DetectorError("fleet hostname denylist contains an invalid entry")
        hostnames.add(candidate)
        if len(hostnames) > MAX_FLEET_HOSTNAMES:
            raise DetectorError(
                f"fleet hostname denylist limit exceeded: {MAX_FLEET_HOSTNAMES} entries"
            )
    return frozenset(hostnames)


def _resolved_file(root: Path, relative_path: str) -> Path | None:
    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        return None
    candidate = root
    for component in PurePosixPath(relative_path).parts:
        candidate /= component
        if candidate.is_symlink():
            return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError:
        return candidate
    if not resolved.is_relative_to(resolved_root):
        return None
    return resolved


@dataclass(frozen=True, slots=True)
class _PendingFinding:
    rule_id: str
    message: str
    start: int


@dataclass(slots=True)
class _WorkLimiter:
    per_file_limit: int
    repository: ScanWorkBudget | None
    per_file_used: int = 0

    def consume(self, amount: int) -> bool:
        if amount < 0:
            raise DetectorError("scan work amount cannot be negative")
        if self.per_file_used + amount > self.per_file_limit:
            return False
        if (
            self.repository is not None
            and self.repository.used + amount > self.repository.max_work_units
        ):
            return False
        self.per_file_used += amount
        if self.repository is not None:
            self.repository.used += amount
        return True


@dataclass(slots=True)
class _TextScan:
    path: str
    text: str
    max_findings: int
    limiter: _WorkLimiter
    pending: list[_PendingFinding]
    seen: set[tuple[str, int]]
    capped: bool = False

    def begin_pass(self) -> bool:
        if self.limiter.consume(len(self.text)):
            return True
        self.capped = True
        return False

    def consume_candidate(self) -> bool:
        if self.limiter.consume(1):
            return True
        self.capped = True
        return False

    def add(self, rule_id: str, message: str, start: int) -> bool:
        identity = (rule_id, start)
        if identity in self.seen:
            return True
        if len(self.pending) >= self.max_findings:
            self.capped = True
            return False
        self.pending.append(_PendingFinding(rule_id, message, start))
        self.seen.add(identity)
        return True


def _regex_findings(
    scan: _TextScan,
    pattern: Pattern[str],
    rule_id: str,
    message: str,
    *,
    start_group: str | None = None,
) -> bool:
    if not scan.begin_pass():
        return False
    for match in pattern.finditer(scan.text):
        if not scan.consume_candidate() or not scan.add(
            rule_id,
            message,
            match.start(start_group) if start_group is not None else match.start(),
        ):
            return False
    return True


def _network_findings(scan: _TextScan) -> bool:
    if not scan.begin_pass():
        return False
    cgnat = ipaddress.ip_network("100.64.0.0/10")
    private_networks = (
        ipaddress.ip_network("10.0.0.0/8"),
        ipaddress.ip_network("172.16.0.0/12"),
        ipaddress.ip_network("192.168.0.0/16"),
    )
    for match in _IPV4_CANDIDATE.finditer(scan.text):
        if not scan.consume_candidate():
            return False
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if address in cgnat:
            if not scan.add(
                RULE_CGNAT_IPV4,
                "CGNAT or Tailscale IPv4 address detected",
                match.start(),
            ):
                return False
        elif any(address in network for network in private_networks):
            if not scan.add(
                RULE_PRIVATE_IPV4,
                "RFC1918 IPv4 address detected",
                match.start(),
            ):
                return False
    return True


def _is_conventional_email(email: str, allow_localhost: bool) -> bool:
    local, domain = email.rsplit("@", 1)
    if (
        not local
        or len(local) > 64
        or re.fullmatch(r"[A-Za-z0-9]+(?:[._+-][A-Za-z0-9]+)*", local) is None
    ):
        return False
    if domain.casefold() == "localhost":
        return allow_localhost
    if "." not in domain or len(domain) > 253:
        return False
    labels = domain.split(".")
    if any(
        not label
        or len(label) > 63
        or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?", label) is None
        for label in labels
    ):
        return False
    return re.fullmatch(r"[A-Za-z](?:[A-Za-z0-9-]*[A-Za-z0-9])?", labels[-1]) is not None


def _email_findings(
    scan: _TextScan,
    public_contacts: frozenset[str],
    detect_localhost_emails: bool,
) -> bool:
    if not scan.begin_pass():
        return False
    example_domains = {"example.com", "example.net", "example.org"}
    for match in _EMAIL.finditer(scan.text):
        if not scan.consume_candidate():
            return False
        email = match.group().casefold()
        domain = email.rsplit("@", 1)[1]
        if (
            not _is_conventional_email(email, detect_localhost_emails)
            or email in public_contacts
            or domain in example_domains
        ):
            continue
        if not scan.add(
            RULE_NONPUBLIC_EMAIL,
            "Email address is not an explicitly public contact",
            match.start(),
        ):
            return False
    return True


def _fleet_findings(scan: _TextScan, fleet_hostnames: frozenset[str]) -> bool:
    for hostname in sorted(fleet_hostnames):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9.-]){re.escape(hostname)}(?![A-Za-z0-9.-])",
            re.IGNORECASE,
        )
        completed = _regex_findings(
            scan,
            pattern,
            RULE_FLEET_HOSTNAME,
            "Configured fleet hostname detected",
        )
        if not completed:
            return False
    return True


def _utf8_width(character: str) -> int:
    codepoint = ord(character)
    if codepoint <= 0x7F:
        return 1
    if codepoint <= 0x7FF:
        return 2
    if codepoint <= 0xFFFF:
        return 3
    return 4


def _materialize_findings(path: str, text: str, pending: list[_PendingFinding]) -> list[Finding]:
    if not pending:
        return []
    ordered = sorted(pending, key=lambda finding: (finding.start, finding.rule_id))
    findings: list[Finding] = []
    pending_index = 0
    line = 1
    byte_offset = 0
    for character_index, character in enumerate(text):
        while pending_index < len(ordered) and ordered[pending_index].start == character_index:
            item = ordered[pending_index]
            findings.append(
                Finding(
                    rule_id=item.rule_id,
                    path=path,
                    message=item.message,
                    line=line,
                    offset=byte_offset,
                )
            )
            pending_index += 1
        byte_offset += _utf8_width(character)
        if character == "\n":
            line += 1
    while pending_index < len(ordered) and ordered[pending_index].start == len(text):
        item = ordered[pending_index]
        findings.append(
            Finding(
                rule_id=item.rule_id,
                path=path,
                message=item.message,
                line=line,
                offset=byte_offset,
            )
        )
        pending_index += 1
    return findings


def _scan_text(
    path: str,
    text: str,
    policy: DetectorPolicy,
    fleet_hostnames: frozenset[str],
    limiter: _WorkLimiter,
) -> list[Finding]:
    scan = _TextScan(
        path=path,
        text=text,
        max_findings=policy.max_findings_per_file,
        limiter=limiter,
        pending=[],
        seen=set(),
    )
    completed = _network_findings(scan)
    if completed:
        completed = _regex_findings(
            scan,
            _POSIX_PATH,
            RULE_POSIX_PATH,
            "Absolute POSIX or macOS path detected",
        )
    if completed:
        completed = _regex_findings(
            scan,
            _STRUCTURED_POSIX_PATH,
            RULE_POSIX_PATH,
            "Absolute POSIX or macOS path detected",
            start_group="path",
        )
    if completed:
        for pattern in (_WINDOWS_DRIVE_PATH, _WINDOWS_UNC_PATH):
            completed = _regex_findings(
                scan,
                pattern,
                RULE_WINDOWS_PATH,
                "Absolute Windows drive or UNC path detected",
            )
            if not completed:
                break
    if completed:
        completed = _fleet_findings(scan, fleet_hostnames)
    if completed:
        completed = _email_findings(
            scan,
            policy.public_contacts,
            policy.detect_localhost_emails,
        )
    if completed:
        completed = _regex_findings(
            scan,
            _RAW_KEY,
            RULE_RAW_BENCHMARK_KEY,
            "Raw benchmark, question, answer, or response key detected",
        )
    if completed:
        completed = _regex_findings(
            scan,
            _COLLABORATION_METADATA,
            RULE_COLLABORATION_METADATA,
            "Slack or Discord metadata detected",
        )
    if completed:
        _regex_findings(
            scan,
            _SESSION_METADATA,
            RULE_SESSION_METADATA,
            "Session metadata detected",
        )
    if not scan.limiter.consume(len(text)):
        scan.capped = True
        scan.pending.clear()
    findings = _materialize_findings(path, text, scan.pending)
    if scan.capped:
        findings.append(
            _path_finding(
                RULE_SCAN_WORK_LIMIT,
                path,
                "Content scan stopped at the configured finding or work limit",
            )
        )
    return findings


def _filter_python_code_context(
    path: str,
    text: str,
    findings: list[Finding],
) -> list[Finding]:
    if not path.endswith((".py", ".pyi")):
        return findings
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return findings

    ignored: list[tuple[int, int, frozenset[str]]] = []

    def add_constants(node: ast.AST, rules: set[str]) -> None:
        frozen = frozenset(rules)
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, (str, bytes))
                and hasattr(child, "end_lineno")
                and child.end_lineno is not None
            ):
                ignored.append((child.lineno, child.end_lineno, frozen))

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            add_constants(node, _PYTHON_CONTEXTUAL_RULES)
        elif isinstance(node, ast.Call):
            function = node.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "re"
                and function.attr == "compile"
            ):
                add_constants(node, {RULE_POSIX_PATH, RULE_WINDOWS_PATH})
            elif (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "ipaddress"
                and function.attr == "ip_network"
            ):
                add_constants(node, {RULE_PRIVATE_IPV4, RULE_CGNAT_IPV4})

    return [
        finding
        for finding in findings
        if finding.line is None
        or not any(
            start <= finding.line <= end and finding.rule_id in rules
            for start, end, rules in ignored
        )
    ]


def _gitleaks_requirement(path: str, status: GitleaksStatus) -> list[Finding]:
    if status is GitleaksStatus.PASSED:
        return []
    return [
        Finding(
            rule_id=RULE_GITLEAKS_REQUIRED,
            path=path,
            message="Credential content scan requires Gitleaks and was not run",
        )
    ]


def _is_binary(raw: bytes) -> bool:
    if raw.startswith(_BINARY_MAGIC):
        return True
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return True
    return any((byte < 32 and byte not in {9, 10, 13}) or byte == 127 for byte in raw)


def scan_file(
    root: Path,
    relative_path: str,
    *,
    policy: DetectorPolicy | None = None,
    fleet_hostnames: frozenset[str] = frozenset(),
    gitleaks_status: GitleaksStatus = GitleaksStatus.NOT_RUN,
    work_budget: ScanWorkBudget | None = None,
) -> list[Finding]:
    """Scan one repository-relative file under fixed memory and input limits."""

    active_policy = policy or DetectorPolicy()
    if not isinstance(gitleaks_status, GitleaksStatus):
        raise DetectorError("Gitleaks status must be an explicit GitleaksStatus value")
    if work_budget is not None and not isinstance(work_budget, ScanWorkBudget):
        raise DetectorError("work_budget must be a ScanWorkBudget")
    if len(fleet_hostnames) > MAX_FLEET_HOSTNAMES:
        raise DetectorError(f"fleet hostname limit exceeded: {MAX_FLEET_HOSTNAMES}")
    if any(
        not isinstance(hostname, str)
        or len(hostname) > MAX_IDENTIFIER_LENGTH
        or _HOSTNAME.fullmatch(hostname) is None
        for hostname in fleet_hostnames
    ):
        raise DetectorError("fleet hostname must be valid and bounded")
    normalized_hostnames = frozenset(hostname.casefold() for hostname in fleet_hostnames)
    if not _is_canonical_relative_path(relative_path):
        path_findings = [
            _path_finding(
                RULE_PATH_ESCAPE,
                relative_path,
                "Path is not canonical repository-relative syntax",
            )
        ]
        path_findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(path_findings)
    if _has_forbidden_directory(relative_path):
        path_findings = [
            _path_finding(
                RULE_FORBIDDEN_DIRECTORY,
                relative_path,
                "Forbidden private, raw-result, or cache directory detected",
            )
        ]
        path_findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(path_findings)

    findings: list[Finding] = []
    limiter = _WorkLimiter(active_policy.max_work_units_per_file, work_budget)
    if _secret_like_filename(relative_path):
        findings.append(
            _path_finding(
                RULE_SECRET_FILENAME,
                relative_path,
                "Secret-like filename detected",
            )
        )

    candidate = _resolved_file(root, relative_path)
    if candidate is None:
        findings.append(
            _path_finding(
                RULE_PATH_ESCAPE,
                relative_path,
                "Path escapes the scan root or contains a symlink",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    if not candidate.is_file():
        raise DetectorError("scan target must be an existing regular file")
    try:
        size = candidate.stat().st_size
    except OSError as error:
        raise DetectorError("scan target metadata cannot be read") from error
    if size > active_policy.max_file_bytes:
        findings.append(
            _path_finding(
                RULE_OVERSIZED_FILE,
                relative_path,
                "File exceeds the configured public-export size limit",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    if size > active_policy.max_scan_bytes:
        findings.append(
            _path_finding(
                RULE_OVERSIZED_FILE,
                relative_path,
                "File exceeds the configured content-scan limit",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    if not limiter.consume(size):
        findings.append(
            _path_finding(
                RULE_SCAN_WORK_LIMIT,
                relative_path,
                "Content scan stopped at the configured finding or work limit",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    try:
        with candidate.open("rb") as stream:
            raw = stream.read(active_policy.max_scan_bytes + 1)
    except OSError as error:
        raise DetectorError("scan target cannot be read") from error
    growth = len(raw) - size
    if growth > 0 and not limiter.consume(growth):
        findings.append(
            _path_finding(
                RULE_SCAN_WORK_LIMIT,
                relative_path,
                "Content scan stopped at the configured finding or work limit",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    if len(raw) > active_policy.max_scan_bytes:
        findings.append(
            _path_finding(
                RULE_OVERSIZED_FILE,
                relative_path,
                "File exceeds the configured content-scan limit",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    if _is_binary(raw):
        findings.append(
            _path_finding(
                RULE_BINARY_FILE,
                relative_path,
                "Binary file detected",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        findings.append(
            _path_finding(
                RULE_BINARY_FILE,
                relative_path,
                "Non-UTF-8 binary file detected",
            )
        )
        findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
        return _sorted_findings(findings)

    findings.extend(
        _filter_python_code_context(
            relative_path,
            text,
            _scan_text(
                relative_path,
                text,
                active_policy,
                normalized_hostnames,
                limiter,
            ),
        )
    )
    findings.extend(_gitleaks_requirement(relative_path, gitleaks_status))
    return _sorted_findings(findings)


def _sorted_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(
        findings,
        key=lambda finding: (
            finding.path,
            finding.offset if finding.offset is not None else -1,
            finding.rule_id,
        ),
    )


def apply_allowlist(findings: list[Finding], entries: list[AllowlistEntry]) -> list[Finding]:
    """Apply exact temporary exceptions, rechecking rules and expiration at use time."""

    if len(entries) > MAX_ALLOWLIST_ENTRIES:
        raise DetectorError(f"allowlist entry limit exceeded: {MAX_ALLOWLIST_ENTRIES}")
    now = datetime.now(UTC)
    for entry in entries:
        _reject_non_allowlistable_rule(entry.rule_id)
        if entry.expires_at <= now:
            raise DetectorError("allowlist entry is expired")
    allowed = {(entry.path, entry.rule_id) for entry in entries}
    return [finding for finding in findings if (finding.path, finding.rule_id) not in allowed]
