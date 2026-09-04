import json

import httpx
import pytest

from model_forge.client import OpenAIClient, Request


@pytest.mark.asyncio
async def test_completion_and_stream_metrics() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("stream"):
            chunks = [
                'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
                'data: {"choices":[{"delta":{"content":"!"}}],"usage":{"completion_tokens":2,"prompt_tokens":4}}\n\n',
                "data: [DONE]\n\n",
            ]
            return httpx.Response(200, text="".join(chunks))
        return httpx.Response(200, json={"choices":[{"message":{"content":None,"reasoning_content":"reasoned answer"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}})

    transport = httpx.MockTransport(handler)
    client = OpenAIClient("http://test/v1", "model", transport=transport)
    result = await client.complete(Request(messages=[{"role":"user","content":"2+2"}], max_tokens=4))
    assert result.text == "reasoned answer" and result.completion_tokens == 2
    streamed = await client.complete(Request(messages=[{"role":"user","content":"hello"}], stream=True))
    assert streamed.text == "Hi!"
    assert streamed.completion_tokens == 2
    assert streamed.ttft_ms is not None and streamed.latency_ms >= streamed.ttft_ms
    assert streamed.end_to_end_tps > 0
