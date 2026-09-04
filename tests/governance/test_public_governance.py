from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote

import pytest
import yaml
from jsonschema import Draft202012Validator, ValidationError

from model_forge.public_export.detectors import GitleaksStatus, scan_file

REPO_ROOT = Path(__file__).resolve().parents[2]
GOVERNANCE_FILES = {
    "SECURITY.md",
    "CONTRIBUTING.md",
    "GOVERNANCE.md",
    "SUPPORT.md",
    "CODE_OF_CONDUCT.md",
}
COMMUNITY_FILES = {
    ".github/CODEOWNERS",
    ".github/pull_request_template.md",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
}
DECISION_FILES = {
    "docs/decisions/0001-private-archive-public-root-separation.md",
    "docs/decisions/0002-release-publication-authority.md",
}
ISSUE_FORM_SCHEMA_FILE = "tests/governance/schemas/github-issue-form.schema.json"
PUBLIC_GOVERNANCE_FILES = (
    GOVERNANCE_FILES | COMMUNITY_FILES | DECISION_FILES | {ISSUE_FORM_SCHEMA_FILE}
)
VERIFIED_GITHUB_OWNERS = {"@HangGlidersRule"}
REQUIRED_CODEOWNER_PATTERNS = {
    "/.github/",
    "/.gitleaks.toml",
    "/SECURITY.md",
    "/contracts/",
    "/containers/",
    "/docs/decisions/",
    "/scripts/",
    "/src/model_forge/public_export/",
    "/tools/public_export/",
    "/models/*/results/publication-readiness-ledger.json",
}
ISSUE_FORM_FILES = (
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
)
ISSUE_FORM_FIELDS = {"name", "description", "title", "body"}
# GitHub supports an `upload` element too, but public forms here stay text-only.
ALLOWED_BODY_TYPES = {"markdown", "input", "textarea", "dropdown", "checkboxes"}
FIELD_VALIDATION_TYPES = {"input", "textarea", "dropdown"}
FIELD_IDENTIFIER = re.compile(r"[a-z][a-z0-9_-]*")
LOCAL_LINK = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
FORBIDDEN_AUTOMATION_MARKERS = {
    "/ai-review",
    "/paid-review",
    "@model-forge-bot",
    "pull_request_target",
    "repository_dispatch",
    "workflow_dispatch",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        assert key not in result, f"duplicate YAML key: {key}"
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(relative: str) -> dict[str, Any]:
    data = yaml.load((REPO_ROOT / relative).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(data, dict)
    return data


class IssueFormError(ValueError):
    """An issue form violates GitHub's form schema or this project's field policy."""


@lru_cache(maxsize=1)
def _issue_form_schema() -> dict[str, Any]:
    schema = json.loads((REPO_ROOT / ISSUE_FORM_SCHEMA_FILE).read_text(encoding="utf-8"))
    assert isinstance(schema, dict)
    Draft202012Validator.check_schema(schema)
    return schema


def _assert_github_form_syntax(form: dict[str, Any], source: str) -> None:
    try:
        Draft202012Validator(_issue_form_schema()).validate(form)
    except ValidationError as error:
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise IssueFormError(
            f"{source} violates the form schema at {location}: {error.message}"
        ) from error


def _assert_checkbox_acknowledgement(field: dict[str, Any], source: str, name: str) -> None:
    options = field["attributes"]["options"]
    if not options:
        raise IssueFormError(f"{source} checkbox field {name} declares no acknowledgement option")
    for position, option in enumerate(options):
        if option.get("required") is not True:
            raise IssueFormError(
                f"{source} checkbox field {name} option {position} "
                "must acknowledge with required: true"
            )


def _validate_issue_form(form: dict[str, Any], source: str) -> set[str]:
    """Validate GitHub form syntax, then this project's per-field-type requirements."""

    _assert_github_form_syntax(form, source)
    if set(form) != ISSUE_FORM_FIELDS:
        raise IssueFormError(f"{source} must declare exactly {sorted(ISSUE_FORM_FIELDS)}")
    if not all(str(form[field]).strip() for field in ISSUE_FORM_FIELDS - {"body"}):
        raise IssueFormError(f"{source} has a blank top-level field")

    names: set[str] = set()
    for position, field in enumerate(form["body"]):
        kind = field["type"]
        if kind not in ALLOWED_BODY_TYPES:
            raise IssueFormError(f"{source} body[{position}] uses forbidden element type {kind}")
        if kind == "markdown":
            continue
        name = field.get("id")
        if not isinstance(name, str) or not FIELD_IDENTIFIER.fullmatch(name):
            raise IssueFormError(f"{source} body[{position}] needs a lowercase identifier")
        if name in names:
            raise IssueFormError(f"{source} reuses field identifier {name}")
        names.add(name)
        if kind in FIELD_VALIDATION_TYPES:
            if field.get("validations", {}).get("required") is not True:
                raise IssueFormError(f"{source} field {name} must set validations.required")
        else:
            _assert_checkbox_acknowledgement(field, source, name)
    return names


def _synthetic_form(body: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": "Synthetic form",
        "description": "Synthetic fixture for issue-form schema regression tests.",
        "title": "[Synthetic]: ",
        "body": body,
    }


def _value_field(kind: str) -> dict[str, Any]:
    attributes: dict[str, Any] = {"label": "Field"}
    if kind == "dropdown":
        attributes["options"] = ["risk:low — a", "risk:medium — b", "risk:high — c"]
    return {
        "type": kind,
        "id": "field",
        "attributes": attributes,
        "validations": {"required": True},
    }


def _checkbox_field() -> dict[str, Any]:
    return {
        "type": "checkboxes",
        "id": "safety",
        "attributes": {
            "label": "Submission safety",
            "options": [{"label": "I removed restricted content.", "required": True}],
        },
    }


def _codeowner_entries() -> list[tuple[str, tuple[str, ...]]]:
    entries: list[tuple[str, tuple[str, ...]]] = []
    for raw_line in (REPO_ROOT / ".github/CODEOWNERS").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        assert len(fields) >= 2, f"CODEOWNERS entry lacks an owner: {line}"
        entries.append((fields[0], tuple(fields[1:])))
    return entries


def _pattern_has_existing_match(pattern: str) -> bool:
    relative = pattern.removeprefix("/")
    if relative == "*":
        return True
    if relative.endswith("/"):
        return (REPO_ROOT / relative).is_dir()
    if any(character in relative for character in "*?"):
        return any(REPO_ROOT.glob(relative))
    return (REPO_ROOT / relative).exists()


def test_required_public_governance_files_exist() -> None:
    missing = sorted(path for path in PUBLIC_GOVERNANCE_FILES if not (REPO_ROOT / path).is_file())
    assert not missing


def test_governance_identity_boundary_and_authority_are_explicit() -> None:
    governance = (REPO_ROOT / "GOVERNANCE.md").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "public canonical" in governance
    assert "Darkstar is the product brand" in governance
    assert "private operator archives" in governance
    assert "least privilege" in governance
    assert "branch protection" in governance.casefold()
    assert "cryptographically signed" in governance
    assert "reproducible" in governance
    assert "Only maintainers" in governance
    assert "raw evaluation questions" in contributing
    assert "fixed denominator" in contributing
    assert "AI-assisted contributions" in contributing
    assert "human contributor remains responsible" in contributing


def test_codeowners_use_verified_owners_and_existing_paths() -> None:
    entries = _codeowner_entries()
    patterns = {pattern for pattern, _ in entries}
    assert REQUIRED_CODEOWNER_PATTERNS <= patterns
    assert all(owner in VERIFIED_GITHUB_OWNERS for _, owners in entries for owner in owners)
    assert all(owner.startswith("@") and "/" not in owner for _, owners in entries for owner in owners)
    missing = sorted(pattern for pattern, _ in entries if not _pattern_has_existing_match(pattern))
    assert not missing


def test_codeowners_syntax_is_bounded_and_unambiguous() -> None:
    for pattern, owners in _codeowner_entries():
        assert pattern == "*" or pattern.startswith("/")
        assert not pattern.startswith(("!", "\\"))
        assert "[" not in pattern and "]" not in pattern
        assert "#" not in pattern
        assert len(owners) == len(set(owners))


def test_local_issue_form_schema_matches_the_documented_dialect() -> None:
    schema = _issue_form_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    element_types = schema["$defs"]["element"]["properties"]["type"]["enum"]
    assert ALLOWED_BODY_TYPES < set(element_types)
    checkboxes = schema["$defs"]["checkboxesElement"]
    assert "validations" not in checkboxes["properties"]
    assert checkboxes["additionalProperties"] is False


def test_issue_forms_have_strict_schema_and_required_risk_evidence() -> None:
    for relative in ISSUE_FORM_FILES:
        form = _load_yaml(relative)
        names = _validate_issue_form(form, relative)
        assert {"risk", "evidence", "safety"} <= names
        risk = next(item for item in form["body"] if item.get("id") == "risk")
        assert {
            str(option).split(" — ", 1)[0]
            for option in risk["attributes"]["options"]
        } == {"risk:low", "risk:medium", "risk:high"}


def test_checkbox_fields_require_every_acknowledgement_option() -> None:
    for relative in ISSUE_FORM_FILES:
        form = _load_yaml(relative)
        checkboxes = [item for item in form["body"] if item["type"] == "checkboxes"]
        assert checkboxes
        for field in checkboxes:
            assert "validations" not in field
            options = field["attributes"]["options"]
            assert options
            assert all(option["required"] is True for option in options)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"validations": {"required": True}}, "violates the form schema"),
        ({"attributes": {"label": "Safety", "options": []}}, "violates the form schema"),
        (
            {"attributes": {"label": "Safety", "options": [{"label": "I acknowledge."}]}},
            "must acknowledge with required: true",
        ),
        (
            {
                "attributes": {
                    "label": "Safety",
                    "options": [{"label": "I acknowledge.", "required": False}],
                }
            },
            "must acknowledge with required: true",
        ),
    ],
)
def test_invalid_checkbox_fields_are_rejected(
    mutation: dict[str, Any], message: str
) -> None:
    form = _synthetic_form([{**_checkbox_field(), **mutation}])
    with pytest.raises(IssueFormError, match=re.escape(message)):
        _validate_issue_form(form, "synthetic.yml")


@pytest.mark.parametrize("kind", sorted(FIELD_VALIDATION_TYPES))
@pytest.mark.parametrize("validations", [None, {"required": False}])
def test_optional_input_dropdown_and_textarea_fields_are_rejected(
    kind: str, validations: dict[str, bool] | None
) -> None:
    field = _value_field(kind)
    if validations is None:
        del field["validations"]
    else:
        field["validations"] = validations
    form = _synthetic_form([field, _checkbox_field()])
    with pytest.raises(IssueFormError, match="must set validations.required"):
        _validate_issue_form(form, "synthetic.yml")


def test_checkbox_style_option_requirement_is_rejected_on_value_fields() -> None:
    field = _value_field("input")
    field["attributes"]["options"] = [{"label": "I acknowledge.", "required": True}]
    with pytest.raises(IssueFormError, match="violates the form schema"):
        _validate_issue_form(_synthetic_form([field]), "synthetic.yml")


def test_markdown_elements_carry_no_identifier_or_validations() -> None:
    for extra in ({"id": "intro"}, {"validations": {"required": True}}):
        element = {"type": "markdown", "attributes": {"value": "Read the policy."}, **extra}
        with pytest.raises(IssueFormError, match="violates the form schema"):
            _validate_issue_form(_synthetic_form([element, _checkbox_field()]), "synthetic.yml")


def test_upload_elements_are_schema_valid_but_forbidden_by_policy() -> None:
    upload = {"type": "upload", "id": "evidence", "attributes": {"label": "Evidence"}}
    form = _synthetic_form([upload, _checkbox_field()])
    Draft202012Validator(_issue_form_schema()).validate(form)
    with pytest.raises(IssueFormError, match="forbidden element type upload"):
        _validate_issue_form(form, "synthetic.yml")


def test_duplicate_field_identifiers_are_rejected() -> None:
    form = _synthetic_form([_value_field("input"), _value_field("textarea")])
    with pytest.raises(IssueFormError, match="reuses field identifier"):
        _validate_issue_form(form, "synthetic.yml")


def test_conforming_synthetic_form_passes_every_field_type() -> None:
    body = [
        {"type": "markdown", "attributes": {"value": "Read the policy."}},
        _value_field("input"),
        {**_value_field("textarea"), "id": "detail"},
        {**_value_field("dropdown"), "id": "risk"},
        _checkbox_field(),
    ]
    assert _validate_issue_form(_synthetic_form(body), "synthetic.yml") == {
        "field",
        "detail",
        "risk",
        "safety",
    }


def test_blank_issues_stay_disabled_only_while_every_form_validates() -> None:
    for relative in ISSUE_FORM_FILES:
        _validate_issue_form(_load_yaml(relative), relative)

    config = _load_yaml(".github/ISSUE_TEMPLATE/config.yml")

    assert config["blank_issues_enabled"] is False


def test_issue_template_configuration_is_safe_and_non_privileged() -> None:
    config = _load_yaml(".github/ISSUE_TEMPLATE/config.yml")
    assert set(config) == {"blank_issues_enabled", "contact_links"}
    assert isinstance(config["contact_links"], list) and config["contact_links"]
    assert all(set(link) == {"name", "url", "about"} for link in config["contact_links"])
    assert all(str(link["url"]).startswith("https://github.com/") for link in config["contact_links"])

    template_text = "\n".join(
        (REPO_ROOT / path).read_text(encoding="utf-8")
        for path in sorted(COMMUNITY_FILES - {".github/CODEOWNERS"})
    ).casefold()
    assert not (FORBIDDEN_AUTOMATION_MARKERS & set(template_text.split()))
    assert all(marker not in template_text for marker in FORBIDDEN_AUTOMATION_MARKERS)
    for form_path in (
        ".github/ISSUE_TEMPLATE/bug.yml",
        ".github/ISSUE_TEMPLATE/feature.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
    ):
        form = _load_yaml(form_path)
        assert not ({"permissions", "jobs", "on", "run", "uses"} & set(form))


def test_security_contact_and_supported_versions_are_valid() -> None:
    security = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    expected = "https://github.com/HangGlidersRule/model-forge/security/advisories/new"
    assert expected in security
    assert "Supported versions" in security
    assert "3 business days" in security
    assert "targets, not guarantees" in security
    assert "mailto:" not in security.casefold()
    assert re.search(r"[\w.+-]+@[\w.-]+", security) is None


def test_license_dco_and_contributor_covenant_attribution() -> None:
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    conduct = (REPO_ROOT / "CODE_OF_CONDUCT.md").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License")
    assert "Developer Certificate of Origin 1.1" in contributing
    assert "Signed-off-by:" in contributing
    assert "MIT License" in contributing
    assert "Contributor Covenant, version 2.1" in conduct
    assert "https://www.contributor-covenant.org/version/2/1/code_of_conduct.html" in conduct


def test_governance_markdown_internal_links_resolve() -> None:
    broken: list[str] = []
    for relative in sorted(GOVERNANCE_FILES | DECISION_FILES | {".github/pull_request_template.md"}):
        path = REPO_ROOT / relative
        text = path.read_text(encoding="utf-8")
        for match in LOCAL_LINK.finditer(text):
            target = match.group(1).strip().split(" ", 1)[0].strip("<>")
            if not target or target.startswith("#") or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
                continue
            local_path = unquote(target.split("#", 1)[0])
            resolved = (path.parent / PurePosixPath(local_path)).resolve()
            if not resolved.exists() or not resolved.is_relative_to(REPO_ROOT.resolve()):
                broken.append(f"{relative} -> {target}")
    assert not broken


def test_public_governance_files_have_no_forbidden_metadata() -> None:
    findings = {
        relative: scan_file(
            REPO_ROOT,
            relative,
            gitleaks_status=GitleaksStatus.PASSED,
        )
        for relative in sorted(PUBLIC_GOVERNANCE_FILES)
    }
    assert not {path: result for path, result in findings.items() if result}


def test_public_export_manifest_includes_every_governance_file() -> None:
    manifest = _load_yaml("tools/public_export/public-files.yaml")
    rules = manifest["rules"]
    assert isinstance(rules, list)

    def matches(source: str, relative: str) -> bool:
        return fnmatchcase(relative, source)

    missing: list[str] = []
    for relative in sorted(PUBLIC_GOVERNANCE_FILES):
        matching = [rule for rule in rules if matches(rule["source"], relative)]
        if not matching:
            missing.append(relative)
            continue
        winning = max(matching, key=lambda rule: int(rule.get("precedence", 0)))
        assert winning["disposition"] in {"copy", "transform"}
        assert winning["public_destination"] == "{source}"
    assert not missing
