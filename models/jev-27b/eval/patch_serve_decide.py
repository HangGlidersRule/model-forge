#!/usr/bin/env python3
"""Make serve_decide.py work with a merged model (no LoRA module).

serve_decide.py hard-requires `--lora-modules jev-decision=<dir>` and reads decision_head.json from
that directory, and sends System 1 read-outs to the LoRA module name. A merged checkpoint has the
LoRA folded into the base, so System 1 is the base model itself. This patch:
  1. Reads the adapter directory from JEV_DECIDE_HEAD_DIR when no --lora-modules is given.
  2. Sends System 1 read-outs to the served model name (JEV_DECIDE_LORA unset -> base model).
It changes nothing when --lora-modules is passed, so the same patched file serves the
unmerged reference too. Prints PATCHED and the two replaced lines.
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
p.write_text(s)
print("PATCHED", p)
