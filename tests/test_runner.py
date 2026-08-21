import json
from pathlib import Path

import httpx
import pytest

from model_forge.runner import Runner


@pytest.mark.asyncio
@pytest.mark.private_source_only
async def test_runner_resumes_and_writes_all_artifacts(tmp_path: Path) -> None:
    calls = 0
    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        content = "4" if "2+2" in str(body["messages"]) else "alpha omega"
        return httpx.Response(200, json={"choices":[{"message":{"content":content},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}})

    spec = tmp_path / "spec.yaml"
    spec.write_text("""
schema_version: '1.0'
name: resume-test
models:
  - {id: base, endpoint: 'http://test/v1', model: base, family: qwen3.6, precision: nvfp4}
tracks:
  - {name: no-think, reasoning: false, temperature: 0, seed: 42}
suites: [smoke]
repeats: 1
""")
    out = tmp_path / "out"
    runner = Runner.from_path(spec, out, transport=httpx.MockTransport(handler))
    await runner.run()
    first_calls = calls
    await runner.run()
    assert calls == first_calls
    for name in ("results.jsonl", "summary.json", "report.md", "provenance.json"):
        assert (out / name).exists()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["models"]["base"]["overall"]["passed"] == 2
