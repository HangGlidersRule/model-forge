from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from model_forge import pr_risk as pr_risk_module
from model_forge.pr_risk import (
    DEFAULT_CONFIG,
    RiskClassificationError,
    changed_paths,
    classify_paths,
    human_summary,
    stable_json,
)

FIXTURES = Path(__file__).parent / "fixtures" / "cases.json"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments), cwd=repository, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _init_repository(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    return path


def _commit(repository: Path, path: str, content: str, message: str) -> str:
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _git(repository, "add", path)
    _git(
        repository,
        "-c",
        "user.name=CI",
        "-c",
        "user.email=security@hangglidersrule.com",
        "commit",
        "-q",
        "-m",
        message,
    )
    return _git(repository, "rev-parse", "HEAD")


def _write_event(path: Path, before: str, after: str) -> Path:
    path.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
    return path


def _write_pull_request_event(path: Path, base: str, head: str) -> Path:
    payload = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("case", json.loads(FIXTURES.read_text(encoding="utf-8")))
def test_classifier_fixtures(case: dict[str, Any]) -> None:
    result = classify_paths(case["paths"])

    assert result["risk"] == case["risk"]
    assert [item["rule"] for item in result["files"]] == case["rules"]


def test_output_is_stable_and_sorted() -> None:
    result = classify_paths(["zeta.txt", "docs/guide.md", "alpha.txt"])

    assert [item["path"] for item in result["files"]] == [
        "alpha.txt",
        "docs/guide.md",
        "zeta.txt",
    ]
    assert stable_json(result).endswith("\n")
    assert stable_json(result) == stable_json(
        classify_paths(["zeta.txt", "docs/guide.md", "alpha.txt"])
    )


def test_matching_is_casefolded_without_changing_reported_paths() -> None:
    paths = [".GITHUB/WORKFLOWS/CI.YML", "security.md", "Docs/Guide.md"]
    result = classify_paths(paths)
    files = {item["path"]: item["rule"] for item in result["files"]}

    assert files == {
        ".GITHUB/WORKFLOWS/CI.YML": "workflows",
        "security.md": "security",
        "Docs/Guide.md": "governance",
    }


def test_summary_contains_only_aggregate_review_guidance() -> None:
    summary = human_summary(classify_paths(["docs/attacker`text.md", "src/model_forge/runner.py"]))

    assert "Overall risk: **medium**" in summary
    assert "high classification does not fail CI" in summary
    assert "attacker" not in summary


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/absolute.py",
        "../escape.py",
        "docs/../escape.py",
        "docs//guide.md",
        "docs\\guide.md",
        "docs/\nsummary.md",
        "a/" * 65 + "file.py",
        "x" * 4097,
    ],
)
def test_invalid_or_unbounded_paths_fail_closed(path: str) -> None:
    with pytest.raises(RiskClassificationError):
        classify_paths([path])


@pytest.mark.parametrize(
    "paths",
    [
        ["README.md", "readme.md"],
        ["docs/café.md", "docs/cafe\u0301.md"],
        ["same.txt", "same.txt"],
    ],
)
def test_unicode_case_and_duplicate_collisions_fail_closed(paths: list[str]) -> None:
    with pytest.raises(RiskClassificationError, match="collision"):
        classify_paths(paths)


def test_too_many_files_fail_closed() -> None:
    with pytest.raises(RiskClassificationError, match="file limit"):
        classify_paths([f"unknown/{index}.txt" for index in range(1001)])


@pytest.mark.parametrize(
    "mutation",
    [
        {"unknown": True},
        {"version": 2},
        {"rules": [{"id": "bad", "risk": "critical", "exact": [], "prefixes": []}]},
    ],
)
def test_unknown_config_fails_closed(mutation: dict[str, object]) -> None:
    config = {**DEFAULT_CONFIG, **mutation}
    with pytest.raises(RiskClassificationError):
        classify_paths(["README.md"], config)


def test_push_event_uses_complete_before_after_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = "1" * 40
    after = "2" * 40
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(
            command, 0, stdout=b"docs/first.md\0src/model_forge/second.py\0", stderr=b""
        )

    monkeypatch.setattr(pr_risk_module.subprocess, "run", fake_run)

    assert changed_paths(event, tmp_path) == [
        "docs/first.md",
        "src/model_forge/second.py",
    ]
    assert commands == [
        ("git", "cat-file", "-e", f"{before}^{{commit}}"),
        ("git", "merge-base", "--is-ancestor", before, after),
        (
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            "-z",
            before,
            after,
        ),
    ]


def test_zero_sha_push_lists_full_three_commit_tree_and_rejects_short_zero_oid(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path / "repository")
    _commit(repository, ".github/workflows/early.yml", "name: early\n", "first")
    _commit(repository, "docs/middle.md", "# Middle\n", "second")
    after = _commit(repository, "README.md", "# Tip\n", "third")
    event = _write_event(tmp_path / "event.json", "0" * 40, after)

    paths = changed_paths(event, repository)
    assert paths == [
        ".github/workflows/early.yml",
        "README.md",
        "docs/middle.md",
    ]
    assert classify_paths(paths)["risk"] == "high"

    _write_event(event, "0", after)
    with pytest.raises(RiskClassificationError, match="valid base"):
        changed_paths(event, repository)


def test_push_with_before_absent_after_checkout_lists_complete_tree(tmp_path: Path) -> None:
    replaced = _init_repository(tmp_path / "replaced")
    before = _commit(replaced, "docs/discarded.md", "# Discarded\n", "discarded root")

    repository = _init_repository(tmp_path / "repository")
    _commit(repository, ".github/workflows/new.yml", "name: new\n", "new root")
    after = _commit(repository, "docs/guide.md", "# Guide\n", "new docs")
    event = _write_event(tmp_path / "event.json", before, after)

    assert changed_paths(event, repository) == [
        ".github/workflows/new.yml",
        "docs/guide.md",
    ]


def test_non_ancestor_force_push_lists_complete_tree_instead_of_diffing(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path / "repository")
    before = _commit(repository, "old-only.txt", "old\n", "superseded root")
    _git(repository, "checkout", "-q", "--orphan", "replacement")
    _git(repository, "rm", "-rq", "--cached", ".")
    (repository / "old-only.txt").unlink()
    after = _commit(repository, ".github/workflows/new.yml", "name: new\n", "replacement root")
    event = _write_event(tmp_path / "event.json", before, after)

    assert _git(repository, "cat-file", "-t", before) == "commit"
    assert changed_paths(event, repository) == [".github/workflows/new.yml"]


def test_normal_push_still_reports_only_the_before_after_difference(tmp_path: Path) -> None:
    repository = _init_repository(tmp_path / "repository")
    before = _commit(repository, "docs/first.md", "# First\n", "first")
    after = _commit(repository, "src/model_forge/second.py", "value = 1\n", "second")
    event = _write_event(tmp_path / "event.json", before, after)

    assert changed_paths(event, repository) == ["src/model_forge/second.py"]


def test_out_of_date_pull_request_compares_the_two_tips_instead_of_the_whole_tree(
    tmp_path: Path,
) -> None:
    repository = _init_repository(tmp_path / "repository")
    _commit(repository, ".github/workflows/ci.yml", "name: ci\n", "workflow")
    split = _commit(repository, "docs/first.md", "# First\n", "shared history")
    trunk = _git(repository, "rev-parse", "--abbrev-ref", "HEAD")
    _git(repository, "checkout", "-q", "-b", "feature", split)
    head = _commit(repository, "src/model_forge/feature.py", "value = 1\n", "feature work")
    _git(repository, "checkout", "-q", trunk)
    base = _commit(repository, "docs/base-advance.md", "# Advance\n", "base moves on")
    event = _write_pull_request_event(tmp_path / "event.json", base, head)

    with pytest.raises(subprocess.CalledProcessError):
        _git(repository, "merge-base", "--is-ancestor", base, head)

    paths = changed_paths(event, repository)
    assert paths == _git(repository, "diff", "--name-only", base, head).splitlines()
    assert paths == ["docs/base-advance.md", "src/model_forge/feature.py"]
    assert classify_paths(paths)["risk"] == "medium"


def test_pull_request_event_diffs_the_range_without_requiring_ancestry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = "1" * 40
    head = "2" * 40
    event = _write_pull_request_event(tmp_path / "event.json", base, head)
    commands: list[tuple[str, ...]] = []

    def fake_run(
        command: tuple[str, ...], **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"docs/first.md\0", stderr=b"")

    monkeypatch.setattr(pr_risk_module.subprocess, "run", fake_run)

    assert changed_paths(event, tmp_path) == ["docs/first.md"]
    assert commands == [
        ("git", "cat-file", "-e", f"{base}^{{commit}}"),
        ("git", "cat-file", "-e", f"{head}^{{commit}}"),
        (
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            "-z",
            base,
            head,
        ),
    ]


@pytest.mark.parametrize(
    ("base_override", "head_override"),
    [
        pytest.param("3" * 40, None, id="absent-base"),
        pytest.param(None, "3" * 40, id="absent-head"),
        pytest.param("0" * 40, None, id="zero-base"),
    ],
)
def test_pull_request_without_both_commit_objects_fails_closed(
    tmp_path: Path, base_override: str | None, head_override: str | None
) -> None:
    repository = _init_repository(tmp_path / "repository")
    base = _commit(repository, "docs/first.md", "# First\n", "base")
    head = _commit(repository, "src/model_forge/second.py", "value = 1\n", "head")
    event = _write_pull_request_event(
        tmp_path / "event.json", base_override or base, head_override or head
    )

    with pytest.raises(RiskClassificationError, match="unavailable"):
        changed_paths(event, repository)
