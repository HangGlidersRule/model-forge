from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Request:
    messages: list[dict[str, Any]]
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 42
    stream: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completion:
    text: str
    latency_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    raw: dict[str, Any]

    @property
    def end_to_end_tps(self) -> float:
        return self.completion_tokens / (self.latency_ms / 1000) if self.latency_ms else 0.0


class OpenAIClient:
    def __init__(self, endpoint: str, model: str, api_key: str = "", *, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 300) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.AsyncClient(transport=transport, timeout=timeout, headers=self.headers)

    async def complete(self, request: Request) -> Completion:
        payload = {"model": self.model, "messages": request.messages, "max_tokens": request.max_tokens, "temperature": request.temperature, "top_p": request.top_p, "seed": request.seed, "stream": request.stream, **request.extra_body}
        started = time.perf_counter()
        if request.stream:
            text, first = "", None
            usage: dict[str, Any] = {}
            raw: dict[str, Any] = {}
            async with self.client.stream("POST", f"{self.endpoint}/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    raw = chunk
                    choices = chunk.get("choices", [])
                    token = choices[0].get("delta", {}).get("content", "") if choices else ""
                    if token and first is None:
                        first = time.perf_counter()
                    text += token or ""
                    usage = chunk.get("usage") or usage
            ended = time.perf_counter()
            return Completion(text, (ended-started)*1000, ((first-started)*1000 if first else None), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), None, raw)
        response = await self.client.post(f"{self.endpoint}/chat/completions", json=payload)
        ended = time.perf_counter()
        response.raise_for_status()
        raw = response.json()
        choice = raw["choices"][0]
        usage = raw.get("usage", {})
        message = choice.get("message", {})
        text = (
            message.get("content")
            or message.get("reasoning_content")
            or message.get("reasoning")
            or ""
        )
        return Completion(text, (ended-started)*1000, None, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), choice.get("finish_reason"), raw)

    async def close(self) -> None:
        await self.client.aclose()
