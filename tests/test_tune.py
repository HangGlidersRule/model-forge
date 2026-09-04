import json
from pathlib import Path

import pytest

from model_forge.cli import parser
from model_forge.recipe import RecipeError, load_recipe
from model_forge.tune import (
    ComposeSpec,
    TuneCandidate,
    TuneMatrix,
    artifact_sha,
    bench_candidate,
    candidate_cache_path,
    render_compose,
    render_markdown,
    run_sweep,
    winner_from,
)


def test_candidate_ordering_keys_and_matrix_defaults():
    matrix = TuneMatrix(
        mtp_min=10,
        drafters=(("dflash", "org/draft model", 4), ("dspark", "org/spark", 8)),
    )
    assert TuneMatrix().mtp_max == 12
    assert [candidate.key for candidate in matrix.candidates()] == [
        "mtp10",
        "mtp11",
        "mtp12",
        "dflash4-org-draft-model",
        "dspark8-org-spark",
    ]


def test_spec_config_shapes():
    assert TuneCandidate("mtp", 11).spec_config() == {
        "method": "mtp",
        "num_speculative_tokens": 11,
    }
    assert TuneCandidate("dflash", 4, "org/draft").spec_config() == {
        "method": "dflash",
        "model": "org/draft",
        "num_speculative_tokens": 4,
    }
    assert TuneCandidate("dspark", 7, "org/spark").spec_config()["method"] == "dspark"


def _lane(mean):
    return {"mean": mean, "min": mean, "max": mean, "n_valid": 2, "skipped": 0}


def test_winner_uses_lane_weights():
    report = {
        "lane_weights": {4: 0.6, 16: 0.3, 48: 0.1},
        "results": {
            "short-fast": {"4": _lane(100), "16": _lane(10), "48": _lane(10)},
            "long-fast": {"4": _lane(10), "16": _lane(100), "48": _lane(100)},
        },
    }
    winner, reason = winner_from(report)
    assert winner == "short-fast"
    assert "weighted" in reason


def test_custom_lanes_get_uniform_weights_and_real_winner():
    matrix = TuneMatrix(mtp_max=1, lanes_k=(8, 32))
    assert matrix.lane_weights == ((8, 0.5), (32, 0.5))
    winner, _ = winner_from(
        {
            "matrix": {"lanes_k": [8, 32]},
            "lane_weights": dict(matrix.lane_weights),
            "results": {"mtp1": {"8": _lane(20), "32": _lane(10)}},
        }
    )
    assert winner == "mtp1"


def test_lane_weights_must_match_lanes():
    with pytest.raises(ValueError, match="not in lanes_k"):
        TuneMatrix(lanes_k=(8, 32), lane_weights=((4, 1.0),))


def test_bench_filters_eos_truncated_runs(monkeypatch):
    replies = iter(
        [
            {"usage": {"completion_tokens": 79}},
            {"usage": {"completion_tokens": 100}},
        ]
    )
    monkeypatch.setattr("model_forge.tune._post_json", lambda *args, **kwargs: next(replies))
    result = bench_candidate(
        "http://example",
        "model",
        TuneMatrix(mtp_max=1, lanes_k=(4,), max_tokens=100, runs=2, warmup=0),
    )["4"]
    assert result.skipped == 1
    assert len(result.tok_s) == 1
    assert result.mean > 0


def test_compose_is_deterministic_and_contains_spec_config():
    spec = ComposeSpec(
        image="runtime:test",
        model_dir="D:/models/a",
        served_name="a",
        candidate=TuneCandidate("mtp", 10),
        port=8103,
        container_name="mf-tune",
    )
    first = render_compose(spec)
    assert first == render_compose(spec)
    assert json.dumps(spec.candidate.spec_config(), sort_keys=True) in first
    assert "VLLM_WSL2_ENABLE_PIN_MEMORY=1" in first
    assert '"8103:8000"' in first


def test_artifact_sha_tracks_json_and_shard_size(tmp_path):
    (tmp_path / "config.json").write_text('{"model":"a"}')
    shard = tmp_path / "model-1.safetensors"
    shard.write_bytes(b"123")
    first = artifact_sha(str(tmp_path))
    assert first == artifact_sha(str(tmp_path))
    shard.write_bytes(b"1234")
    assert artifact_sha(str(tmp_path)) != first


def test_run_sweep_resumes_cached_candidate_without_boot(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    results = tmp_path / "results"
    candidate = TuneCandidate("mtp", 1)
    matrix = TuneMatrix(mtp_max=1, lanes_k=(4,), lane_weights=((4, 1.0),))
    cached = candidate_cache_path(results, artifact_sha(str(artifact)), candidate, matrix)
    cached.parent.mkdir(parents=True)
    cached.write_text(json.dumps({"4": _lane(42)}))

    def unexpected_boot(*args, **kwargs):
        raise AssertionError("cached candidate must not boot")

    monkeypatch.setattr("model_forge.tune.boot_and_wait", unexpected_boot)
    report = run_sweep(
        artifact_dir=str(artifact),
        served_name="model",
        image="runtime:test",
        matrix=matrix,
        results_dir=results,
        host="unreachable.invalid",
        user="devin",
        key="/tmp/key",
    )
    assert report["winner"] == "mtp1"
    assert report["results"]["mtp1"]["4"]["mean"] == 42
    assert (cached.parent / "report.md").is_file()


def test_run_sweep_does_not_reuse_cache_when_runs_change(monkeypatch, tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "config.json").write_text("{}")
    results = tmp_path / "results"
    candidate = TuneCandidate("mtp", 1)
    first_matrix = TuneMatrix(mtp_max=1, lanes_k=(4,), runs=1)
    first_cache = candidate_cache_path(
        results, artifact_sha(str(artifact)), candidate, first_matrix
    )
    first_cache.parent.mkdir(parents=True)
    first_cache.write_text(json.dumps({"4": _lane(42)}))

    boots = []
    monkeypatch.setattr(
        "model_forge.tune.boot_and_wait", lambda *args, **kwargs: boots.append(True)
    )
    monkeypatch.setattr(
        "model_forge.tune._benchmark_round",
        lambda *args, **kwargs: {"4": _lane(43)},
    )
    monkeypatch.setattr("model_forge.tune.Remote.down", lambda *args, **kwargs: None)
    second_matrix = TuneMatrix(mtp_max=1, lanes_k=(4,), runs=2)
    report = run_sweep(
        artifact_dir=str(artifact),
        served_name="model",
        image="runtime:test",
        matrix=second_matrix,
        results_dir=results,
        host="unreachable.invalid",
        user="devin",
        key="/tmp/key",
    )

    second_cache = candidate_cache_path(
        results, artifact_sha(str(artifact)), candidate, second_matrix
    )
    assert boots == [True]
    assert first_cache != second_cache
    assert report["results"]["mtp1"]["4"]["mean"] == 43


def test_markdown_lane_columns_are_dynamic():
    report = {
        "served_name": "m",
        "artifact_sha": "sha",
        "image": "image",
        "winner": "mtp1",
        "winner_reason": "weighted",
        "matrix": {"lanes_k": [2, 8], "max_tokens": 32},
        "results": {"mtp1": {"2": _lane(1), "8": _lane(2)}},
        "failed": {},
    }
    assert "| candidate | 2K | 8K |" in render_markdown(report)


def test_tune_cli_help_and_required_arguments():
    tune = next(action for action in parser()._actions if action.dest == "command")
    assert "tune" in tune.choices
    with pytest.raises(SystemExit):
        parser().parse_args(["tune", "--image", "runtime:test"])
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(["tune", "--help"])
    assert exc.value.code == 0
    args = parser().parse_args(
        [
            "tune",
            "--artifact-dir",
            ".",
            "--image",
            "runtime:test",
            "--remote-host",
            "example.invalid",
            "--lanes",
            "8,32",
            "--lane-weights",
            "8:0.5,32:0.5",
        ]
    )
    assert args.lane_weights == ((8, 0.5), (32, 0.5))


def test_dspark_recipe_is_valid_with_required_drafter(tmp_path):
    path = Path(tmp_path) / "dspark.yaml"
    path.write_text(
        """
schema_version: "2.0"
name: dspark-test
family: qwen
source:
  model_id: org/model
  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
validation:
  max_refusal_leakage: 0.1
runtime:
  kv_dtype: bf16
  context_length: 8192
  spec_decode: dspark
  drafter_model: org/spark
  drafter_tokens: 7
outputs:
  artifact_kind: bf16
  publication:
    github: org/repo
"""
    )
    assert load_recipe(path).runtime.spec_decode == "dspark"
    path.write_text(path.read_text().replace("  drafter_model: org/spark\n", ""))
    with pytest.raises(RecipeError, match="drafter_model"):
        load_recipe(path)
