#!/usr/bin/env python3
"""Make serve_decide.py work with a merged model (no LoRA module), and report prefix-cache accounting.

serve_decide.py hard-requires `--lora-modules jev-decision=<dir>` and reads decision_head.json from
that directory, and sends System 1 read-outs to the LoRA module name. A merged checkpoint has the
LoRA folded into the base, so System 1 is the base model itself. This patch:
  1. Reads the adapter directory from JEV_DECIDE_HEAD_DIR when no --lora-modules is given.
  2. Sends System 1 read-outs to the served model name (JEV_DECIDE_LORA unset -> base model).
  3. Issue #23: sums usage.prompt_tokens_details.cached_tokens across the internal calls and emits it in the
     /v1/decide usage block when the server reports it (vLLM --enable-prompt-tokens-details). Without that flag the
     response shape is unchanged.
Steps 1 and 2 change nothing when --lora-modules is passed, so the same patched file serves the
unmerged reference too. Prints PATCHED.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
s = p.read_text()
old1 = '''    path = next((m.path for m in (args.lora_modules or []) if m.name == LORA), None)
    if path is None:
        raise SystemExit(f"serve_decide: start vLLM with --lora-modules {LORA}=<adapter dir>")'''
new1 = '''    path = next((m.path for m in (args.lora_modules or []) if m.name == LORA), None)
    if path is None and os.environ.get("JEV_DECIDE_HEAD_DIR"):
        # Merged checkpoint: no LoRA module; System 1 is the base model (see JEV_DECIDE_MERGED).
        path = os.environ["JEV_DECIDE_HEAD_DIR"]
    if path is None:
        raise SystemExit(f"serve_decide: start vLLM with --lora-modules {LORA}=<adapter dir>")'''
assert s.count(old1) == 1, "setup() anchor not found"
s = s.replace(old1, new1)
old2 = "    lp = await _logprobs(ctx, content, has_img, ids, LORA, allowed=_readout_allowed(ids))\n    return _softmax([(max(lp.get(t, -1e9), -1e9) + b) / temps[kind] for t, b in zip(ids, bias)])"
new2 = "    s1_model = S[\"model\"] if os.environ.get(\"JEV_DECIDE_MERGED\") else LORA\n    lp = await _logprobs(ctx, content, has_img, ids, s1_model, allowed=_readout_allowed(ids))\n    return _softmax([(max(lp.get(t, -1e9), -1e9) + b) / temps[kind] for t, b in zip(ids, bias)])"
assert s.count(old2) == 1, "s1_pass anchor not found"
s = s.replace(old2, new2)

# Issue #23: carry prefix-cache accounting. Ctx also sums usage.prompt_tokens_details.cached_tokens, and both /v1/decide
# usage blocks emit it, but only when the server reported details (vLLM --enable-prompt-tokens-details), so a server
# started without that flag returns exactly today's usage shape.
old3 = '''        self.raw, self.requests, self.prompt_tokens, self.completion_tokens = raw, 0, 0, 0

    def count(self, out):
        self.requests += 1
        if getattr(out, "usage", None):
            self.prompt_tokens += out.usage.prompt_tokens or 0
            self.completion_tokens += out.usage.completion_tokens or 0'''
new3 = '''        self.raw, self.requests, self.prompt_tokens, self.completion_tokens = raw, 0, 0, 0
        self.cached_tokens = None  # None until a server response reports prompt_tokens_details

    def count(self, out):
        self.requests += 1
        if getattr(out, "usage", None):
            self.prompt_tokens += out.usage.prompt_tokens or 0
            self.completion_tokens += out.usage.completion_tokens or 0
            details = getattr(out.usage, "prompt_tokens_details", None)
            if details is not None:
                self.cached_tokens = (self.cached_tokens or 0) + (getattr(details, "cached_tokens", None) or 0)

    def usage(self, **extra):
        u = {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens, **extra}
        if self.cached_tokens is not None:
            u["prompt_tokens_details"] = {"cached_tokens": self.cached_tokens}
        return u'''
assert s.count(old3) == 1, "Ctx anchor not found"
s = s.replace(old3, new3)
old4 = '"usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens},'
new4 = '"usage": ctx.usage(),'
assert s.count(old4) == 1, "system-2 usage anchor not found"
s = s.replace(old4, new4)
old5 = '''"usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
                      "total_tokens": ctx.prompt_tokens + ctx.completion_tokens},'''
new5 = '''"usage": ctx.usage(total_tokens=ctx.prompt_tokens + ctx.completion_tokens),'''
assert s.count(old5) == 1, "main usage anchor not found"
s = s.replace(old5, new5)
p.write_text(s)
print("PATCHED", p)
