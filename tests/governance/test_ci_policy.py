from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github/workflows/ci.yml"
CHECKOUT = "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"
SETUP_PYTHON = "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065"
UPLOAD_ARTIFACT = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
GITLEAKS_IMAGE = (
    "zricethezav/gitleaks"
    "@sha256:c00b6bd0aeb3071cbcb79009cb16a60dd9e0a7c60e2be9ab65d25e6bc8abbb7f"
)
ACTION_PIN = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
CONTAINER_PIN = re.compile(r"^docker://[^@\s]+@sha256:[0-9a-f]{64}$")
EXPRESSION = re.compile(r"\$\{\{\s*(.*?)\s*\}\}")
ALLOWED_EXPRESSIONS = {
    "github.workflow",
    "github.event.pull_request.number || github.ref",
    "matrix.python-version",
    "github.run_id",
}


class _UniqueBaseLoader(yaml.BaseLoader):
    pass


def _construct_unique_mapping(
    loader: yaml.BaseLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        assert isinstance(key, str)
        assert key not in mapping, f"duplicate workflow key: {key}"
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueBaseLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _workflow() -> dict[str, Any]:
    loaded = yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"), Loader=_UniqueBaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        step
        for job in workflow["jobs"].values()
        for step in job["steps"]
    ]


def test_workflow_is_valid_bounded_yaml_with_safe_events_and_permissions() -> None:
    workflow = _workflow()

    assert set(workflow["on"]) == {"push", "pull_request"}
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"]["cancel-in-progress"] == "true"
    assert "self-hosted" not in WORKFLOW_PATH.read_text(encoding="utf-8")


def test_actions_and_container_are_immutable_release_pins() -> None:
    actions = [step["uses"] for step in _steps(_workflow()) if "uses" in step]
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert actions
    assert all(
        ACTION_PIN.fullmatch(action) or CONTAINER_PIN.fullmatch(action)
        for action in actions
    )
    assert {CHECKOUT, SETUP_PYTHON, UPLOAD_ARTIFACT} <= set(actions)
    assert GITLEAKS_IMAGE in workflow_text
    for release in ("v4.4.0", "v5.6.0", "v4.6.2", "v8.30.1"):
        assert f"# {release}" in workflow_text


def test_matrix_and_required_checks_are_deterministic() -> None:
    workflow = _workflow()
    quality = workflow["jobs"]["quality"]
    commands = "\n".join(step.get("run", "") for step in quality["steps"])

    assert quality["strategy"] == {
        "fail-fast": "false",
        "matrix": {"python-version": ["3.11", "3.12"]},
    }
    assert "cache" in next(
        step["with"]
        for step in quality["steps"]
        if step.get("uses") == SETUP_PYTHON
    )
    for command in (
        "ruff check .",
        "mypy src",
        "pytest -q tests/public_export/test_public_file_manifest.py tests/governance",
        "pytest -q",
        'pytest -q -m "not private_source_only"',
        "git diff --check",
    ):
        assert command in commands
    full_pytest = next(
        step for step in quality["steps"] if step.get("name") == "Full pytest"
    )["run"]
    assert "if [[ -e PUBLIC_EXPORT_MANIFEST.json ]]" in full_pytest
    assert "_validate_manifest(m._load_manifest(), m._git_tracked_files())" in full_pytest


def test_expressions_and_shell_steps_do_not_interpolate_attacker_text() -> None:
    workflow_text = WORKFLOW_PATH.read_text(encoding="utf-8")
    expressions = set(EXPRESSION.findall(workflow_text))

    assert expressions == ALLOWED_EXPRESSIONS
    assert all("${{" not in step.get("run", "") for step in _steps(_workflow()))
    assert "secrets." not in workflow_text
    assert "pull_request_target" not in workflow_text
    assert "issue_comment" not in workflow_text


def test_classifier_only_publishes_summary_and_artifact() -> None:
    workflow = _workflow()
    risk_job = workflow["jobs"]["pr-risk"]
    text = str(risk_job).casefold()

    assert "model_forge.pr_risk" in text
    assert "github_step_summary" in text
    assert UPLOAD_ARTIFACT in text
    for mutation in ("label", "comment", "approve", "merge", "dispatch", "pull_request_target"):
        assert mutation not in text


def test_gitleaks_uses_sanitized_copy_and_trusted_policy_outside_checkout() -> None:
    workflow = _workflow()
    steps = workflow["jobs"]["secrets"]["steps"]
    scan = next(step for step in steps if step.get("name") == "Gitleaks")
    scan_run = scan["run"]

    assert 'trusted_dir="$RUNNER_TEMP/gitleaks"' in scan_run
    assert 'scan_root="$RUNNER_TEMP/gitleaks-scan"' in scan_run
    assert 'rm -rf -- "$trusted_dir" "$scan_root"' in scan_run
    assert "'[extend]' 'useDefault = true'" in scan_run
    assert ': > "$trusted_dir/.gitleaksignore"' in scan_run
    assert 'cp -a -- "$GITHUB_WORKSPACE/." "$scan_root/"' in scan_run
    assert (
        'rm -rf -- "$scan_root/.gitleaks.toml" "$scan_root/.gitleaksignore"'
        in scan_run
    )
    assert '--volume "$scan_root:/repo:ro"' in scan_run
    assert '--volume "$trusted_dir:/trusted:ro"' in scan_run
    assert ":/repo/.gitleaks.toml" not in scan_run
    assert ":/repo/.gitleaksignore" not in scan_run
    assert "--config /trusted/config.toml" in scan_run
    assert "--gitleaks-ignore-path /trusted/.gitleaksignore" in scan_run
    assert "$GITHUB_WORKSPACE/.gitleaks.toml" not in scan_run
    assert "$GITHUB_WORKSPACE/.gitleaksignore" not in scan_run
    assert workflow["jobs"]["secrets"]["steps"][0]["with"]["fetch-depth"] == "0"
