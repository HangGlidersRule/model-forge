"""Issue #23: serve_decide.py's /v1/decide must carry prefix-cache accounting (usage.prompt_tokens_details.cached_tokens).

The published serve_decide.py is autotrust's file plus two patch steps kept in this repository
(models/jev-27b/eval/patch_serve_decide.py). These tests apply the patch to a minimal fixture that reproduces the original
file's anchors, then exercise the patched Ctx and usage block without vLLM.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / "models" / "jev-27b" / "eval" / "patch_serve_decide.py"

# The anchors below are copied verbatim from autotrust/JEV-27B@14b425269ed2 serve_decide.py.
FIXTURE = '''import os

LORA = "jev-decision"
S = {"model": "jev"}


def setup(args):
    path = next((m.path for m in (args.lora_modules or []) if m.name == LORA), None)
    if path is None:
        raise SystemExit(f"serve_decide: start vLLM with --lora-modules {LORA}=<adapter dir>")
    return path


async def _logprobs(ctx, content, has_img, ids, model, allowed=None):
    return {}


def _softmax(xs):
    return xs


def _readout_allowed(ids):
    return ids


async def s1_pass(ctx, content, has_img, ids, kind, temps, bias):
    lp = await _logprobs(ctx, content, has_img, ids, LORA, allowed=_readout_allowed(ids))
    return _softmax([(max(lp.get(t, -1e9), -1e9) + b) / temps[kind] for t, b in zip(ids, bias)])


class Ctx:
    def __init__(self, raw):
        self.raw, self.requests, self.prompt_tokens, self.completion_tokens = raw, 0, 0, 0

    def count(self, out):
        self.requests += 1
        if getattr(out, "usage", None):
            self.prompt_tokens += out.usage.prompt_tokens or 0
            self.completion_tokens += out.usage.completion_tokens or 0


def usage_system2(ctx):
    return {"usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens},
            "num_model_requests": ctx.requests}


def usage_main(ctx):
    return {"usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
                      "total_tokens": ctx.prompt_tokens + ctx.completion_tokens},
            "num_model_requests": ctx.requests}
'''


def _patched_module(tmp_path: Path) -> types.ModuleType:
    target = tmp_path / "serve_decide.py"
    target.write_text(FIXTURE, encoding="utf-8")
    proc = subprocess.run([sys.executable, str(PATCH), str(target)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    spec = importlib.util.spec_from_file_location("serve_decide_patched", target)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _out(prompt: int, completion: int, cached: int | None) -> types.SimpleNamespace:
    details = None if cached is None else types.SimpleNamespace(cached_tokens=cached)
    return types.SimpleNamespace(usage=types.SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=details))


def test_cached_tokens_are_summed_and_emitted_when_the_server_reports_them(tmp_path: Path) -> None:
    m = _patched_module(tmp_path)
    ctx = m.Ctx(None)
    ctx.count(_out(100, 5, 64))
    ctx.count(_out(120, 1, 96))
    main = m.usage_main(ctx)["usage"]
    assert main["prompt_tokens"] == 220 and main["completion_tokens"] == 6 and main["total_tokens"] == 226
    assert main["prompt_tokens_details"] == {"cached_tokens": 160}
    assert m.usage_system2(ctx)["usage"]["prompt_tokens_details"] == {"cached_tokens": 160}


def test_response_shape_is_unchanged_when_the_server_does_not_report_details(tmp_path: Path) -> None:
    """A server started without --enable-prompt-tokens-details keeps today's exact usage block."""
    m = _patched_module(tmp_path)
    ctx = m.Ctx(None)
    ctx.count(_out(100, 5, None))
    assert m.usage_main(ctx)["usage"] == {"prompt_tokens": 100, "completion_tokens": 5, "total_tokens": 105}
    assert m.usage_system2(ctx)["usage"] == {"prompt_tokens": 100, "completion_tokens": 5}


def test_a_reported_zero_is_emitted(tmp_path: Path) -> None:
    """cached_tokens 0 from a details-enabled server is a real measurement, not a missing one."""
    m = _patched_module(tmp_path)
    ctx = m.Ctx(None)
    ctx.count(_out(100, 5, 0))
    assert m.usage_main(ctx)["usage"]["prompt_tokens_details"] == {"cached_tokens": 0}


def test_merged_model_patch_still_applies(tmp_path: Path) -> None:
    """The original two merged-checkpoint changes still land alongside the usage fix."""
    _patched_module(tmp_path)
    text = (tmp_path / "serve_decide.py").read_text(encoding="utf-8")
    assert 'os.environ.get("JEV_DECIDE_HEAD_DIR")' in text
    assert 's1_model = S["model"] if os.environ.get("JEV_DECIDE_MERGED") else LORA' in text
