from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = ROOT / "contracts/ai-review/v1"


def _load(name: str) -> dict[str, object]:
    return json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))


def _request() -> dict[str, object]:
    return {
        "schema": "model-forge-ai-review-request/v1",
        "request_id": "018f47d2-7b52-7c91-8f20-a86c77f58d01",
        "repository": "HangGlidersRule/model-forge",
        "pull_request": 42,
        "head_sha": "a" * 40,
        "requested_by": "maintainer",
        "provider": "example-provider",
        "model": "review-model-1",
        "max_input_tokens": 8000,
        "max_output_tokens": 2000,
        "max_cost_microusd": 250000,
        "untrusted_text": ["PR title", "PR body", "patch text"],
    }


def test_request_and_result_examples_validate() -> None:
    request = _request()
    jsonschema.Draft202012Validator(_load("request.schema.json")).validate(request)
    result = {
        "schema": "model-forge-ai-review-result/v1",
        "request_id": request["request_id"],
        "repository": request["repository"],
        "pull_request": request["pull_request"],
        "head_sha": request["head_sha"],
        "request_sha256": "b" * 64,
        "dedupe_key": "c" * 64,
        "status": "completed",
        "advisory": True,
        "provider": request["provider"],
        "model": request["model"],
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "cost_microusd": 10000,
        "findings": [
            {
                "severity": "warning",
                "path": "src/example.py",
                "line": 7,
                "message": "Check this edge case.",
            }
        ],
    }
    jsonschema.Draft202012Validator(_load("result.schema.json")).validate(result)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("head_sha", "main"),
        ("repository", "../private"),
        ("pull_request", 0),
        ("requested_by", ""),
        ("provider", "provider with spaces"),
        ("max_cost_microusd", -1),
    ],
)
def test_request_schema_rejects_unbound_or_malformed_values(
    field: str, value: object
) -> None:
    request = _request()
    request[field] = value
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load("request.schema.json")).validate(request)


def test_request_schema_is_fail_closed_and_bounds_untrusted_text() -> None:
    request = _request()
    request["extra"] = "ignored?"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load("request.schema.json")).validate(request)

    request = _request()
    request["untrusted_text"] = ["x" * 65537]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load("request.schema.json")).validate(request)


@pytest.mark.parametrize(
    "path",
    [
        "a" * 1025,
        "/absolute/review.py",
        "/Users/maintainer/private.py",
        "/Volumes/private/review.py",
        "C:/private/review.py",
        "//server/share/review.py",
        r"\\server\share\review.py",
        "~/.ssh/id_rsa",
        "../private/review.py",
        "src/../private/review.py",
        "src//review.py",
        "src/./review.py",
        r"src\review.py",
    ],
)
def test_result_schema_rejects_noncanonical_finding_paths(path: str) -> None:
    request = _request()
    result = {
        "schema": "model-forge-ai-review-result/v1",
        "request_id": request["request_id"],
        "repository": request["repository"],
        "pull_request": request["pull_request"],
        "head_sha": request["head_sha"],
        "request_sha256": "b" * 64,
        "dedupe_key": "c" * 64,
        "status": "completed",
        "advisory": True,
        "provider": request["provider"],
        "model": request["model"],
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "cost_microusd": 10000,
        "findings": [
            {
                "severity": "warning",
                "path": path,
                "line": 7,
                "message": "Check this edge case.",
            }
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(_load("result.schema.json")).validate(result)


def test_public_ci_has_no_ai_trigger_or_secret_boundary() -> None:
    workflow_path = ROOT / ".github/workflows/ci.yml"
    workflow = yaml.load(workflow_path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    text = workflow_path.read_text(encoding="utf-8").casefold()

    # Deterministic jobs on push/pull_request plus Phase H1 weekly scheduled
    # security sweep. Paid AI review must never be triggered by these events.
    assert set(workflow["on"]) >= {"push", "pull_request"}
    assert set(workflow["on"]) <= {"push", "pull_request", "schedule"}
    assert workflow["permissions"] == {"contents": "read"}
    for forbidden in (
        "pull_request_target",
        "issue_comment",
        "workflow_dispatch",
        "repository_dispatch",
        "api.openai.com",
        "api.anthropic.com",
        "secrets.",
    ):
        assert forbidden not in text
