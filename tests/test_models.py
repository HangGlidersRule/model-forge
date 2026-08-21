from pathlib import Path

import pytest

from model_forge.models import BakeoffSpec, load_spec


def test_load_spec_resolves_env_without_persisting_secret(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_TOKEN", "secret-value")
    path = tmp_path / "spec.yaml"
    path.write_text(
        """
schema_version: '1.0'
name: test
models:
  - id: baseline
    endpoint: http://localhost:8000/v1
    model: qwen
    api_key_env: TEST_TOKEN
    family: qwen3.6
    precision: nvfp4
tracks:
  - name: no-think
    reasoning: false
    temperature: 0
    seed: 42
suites: [smoke]
output:
  retain_prompts: false
"""
    )
    spec = load_spec(path)
    assert spec.models[0].api_key.get_secret_value() == "secret-value"
    dumped = spec.model_dump_json()
    assert "secret-value" not in dumped
    assert spec.schema_version == "1.0"


def test_spec_rejects_duplicate_model_ids() -> None:
    payload = {
        "schema_version": "1.0",
        "name": "x",
        "models": [
            {"id": "same", "endpoint": "http://a/v1", "model": "a", "family": "qwen3.6", "precision": "nvfp4"},
            {"id": "same", "endpoint": "http://b/v1", "model": "b", "family": "qwen3.8", "precision": "fp8"},
        ],
        "tracks": [{"name": "no-think", "reasoning": False}],
        "suites": ["smoke"],
    }
    with pytest.raises(ValueError, match="duplicate model id"):
        BakeoffSpec.model_validate(payload)
