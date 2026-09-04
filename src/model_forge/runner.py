from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from .client import OpenAIClient, Request
from .corpus import load_builtin_suite
from .models import BakeoffSpec, load_spec
from .scoring import evaluate


class Runner:
    def __init__(self, spec: BakeoffSpec, output: Path, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.spec, self.output, self.transport = spec, output, transport

    @classmethod
    def from_path(cls, spec: Path, output: Path, *, transport: httpx.AsyncBaseTransport | None = None) -> "Runner":
        return cls(load_spec(spec), output, transport=transport)

    def _existing(self) -> set[str]:
        path = self.output / "results.jsonl"
        if not path.exists():
            return set()
        return {json.loads(line)["run_id"] for line in path.read_text().splitlines() if line.strip()}

    def _redact(self, text: str) -> str | None:
        if not self.spec.output.retain_responses:
            return None
        for pattern in self.spec.output.redact_patterns:
            text = re.sub(pattern, "[REDACTED]", text)
        return text

    async def run(self) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        cases = [case for suite in self.spec.suites for case in load_builtin_suite(suite)]
        existing = self._existing()
        result_path = self.output / "results.jsonl"
        for target in self.spec.models:
            client = OpenAIClient(target.endpoint, target.model, target.api_key.get_secret_value(), transport=self.transport)
            try:
                for track in self.spec.tracks:
                    for case in cases:
                        for repeat in range(self.spec.repeats):
                            run_id = hashlib.sha256(f"{target.id}|{track.name}|{case.id}|{repeat}".encode()).hexdigest()[:24]
                            if run_id in existing:
                                continue
                            messages = [{"role": "user", "content": case.prompt}]
                            extra: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": track.reasoning}}
                            if track.reasoning_effort:
                                extra["reasoning_effort"] = track.reasoning_effort
                            completion = await client.complete(Request(messages=messages, max_tokens=case.max_tokens, temperature=track.temperature, top_p=track.top_p, seed=track.seed, extra_body=extra))
                            score = evaluate(case, completion.text)
                            record = {"schema_version":"1.0","run_id":run_id,"model_id":target.id,"family":target.family,"precision":target.precision,"track":track.name,"case_id":case.id,"suite":case.suite,"repeat":repeat,"passed":score.passed,"score":score.score,"detail":score.detail,"response":self._redact(completion.text),"prompt":case.prompt if self.spec.output.retain_prompts else None,"latency_ms":completion.latency_ms,"ttft_ms":completion.ttft_ms,"prompt_tokens":completion.prompt_tokens,"completion_tokens":completion.completion_tokens,"end_to_end_tps":completion.end_to_end_tps,"finish_reason":completion.finish_reason}
                            with result_path.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True)+"\n")
            finally:
                await client.close()
        self._write_reports()
        return self.output

    def _write_reports(self) -> None:
        records = [json.loads(line) for line in (self.output / "results.jsonl").read_text().splitlines() if line.strip()]
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for row in records:
            grouped[row["model_id"]][row["suite"]].append(row)
        models: dict[str, Any] = {}
        for model, suites in grouped.items():
            flat = [row for rows in suites.values() for row in rows]
            categories = {}
            for name, rows in suites.items():
                categories[name] = {
                    "passed": sum(row["passed"] for row in rows),
                    "total": len(rows),
                    "accuracy": sum(row["score"] for row in rows) / len(rows),
                    "mean_end_to_end_tps": sum(row["end_to_end_tps"] for row in rows) / len(rows),
                }
            models[model] = {
                "overall": {
                    "passed": sum(row["passed"] for row in flat),
                    "total": len(flat),
                    "accuracy": sum(row["score"] for row in flat) / len(flat),
                },
                "categories": categories,
            }
        summary = {
            "schema_version": "1.0",
            "name": self.spec.name,
            "models": models,
            "note": "Category metrics remain separate; no single score should replace inspection.",
        }
        (self.output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        lines = [
            f"# {self.spec.name}",
            "",
            "Bifrost-style TPS is completion tokens divided by total request latency; "
            "it is not engine aggregate throughput.",
            "",
        ]
        for model, data in models.items():
            overall = data["overall"]
            lines += [
                f"## {model}",
                f"- Overall: {overall['passed']}/{overall['total']} ({overall['accuracy']:.1%})",
            ]
            for name, category in data["categories"].items():
                lines.append(
                    f"- {name}: {category['accuracy']:.1%}; "
                    f"mean E2E {category['mean_end_to_end_tps']:.1f} tok/s"
                )
            lines.append("")
        (self.output / "report.md").write_text("\n".join(lines))
        provenance = {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "spec_sha256": hashlib.sha256(self.spec.model_dump_json().encode()).hexdigest(),
            "models": [model.model_dump(exclude={"api_key"}) for model in self.spec.models],
        }
        (self.output / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        )
