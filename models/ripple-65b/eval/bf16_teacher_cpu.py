#!/usr/bin/env python3
"""Ripple-65B Preview (#39): the BF16 teacher for the KL / top-1 diagnostic, on CPU.

The checkpoint's own remote-code modules (modeling_ricursive.py, sha256 pinned) under transformers 5.18, in FP32 (the
BF16 weights upcast), one decoder layer at a time: each layer is built on the meta device, gets its tensors from the
safetensors shards (strict state-dict load) and runs every prompt before the next layer is read, so the 130 GB model is
never resident. embed_scale, the value residual (the scaled embeddings), the final norm and the untied lm_head follow
RicursiveModel.forward / RicursiveForCausalLM.forward exactly; no attention mask is passed (prompts are unpadded), as
the model's own forward does for an all-ones mask.

Order of work:
  1. Known-answer self-check through this same path: the next token after "The capital of France is" must be " Paris"
     and after "The chemical symbol for gold is" " Au" (top-1). Anything else stops the run (TEACHER-FAILED).
  2. The 64 held-out prompts (ripple39_heldout_64.jsonl from build_ripple_calib.py, token ids as built), all 80
     layers, then the norm and lm_head.
Output (kl_probe.py's collect format, so its unchanged `compare` reads it): {"model", "rows": [prompt][position]} where
position j (0..T-2) is {token id (str): logprob} for the next token after ids[:j+1]: the top 20 plus the actual next
token ids[j+1] (the shape of vLLM's prompt_logprobs=20 entry for prompt token j+1). Also "prompts": source, file, index,
token count and span_start per prompt, and "heldout_sha256".
Resume: the hidden states after every 4th layer go to WORK/state.pt (atomic rename) with the layer index; a restart
with the same inputs continues from there. Progress: WORK/progress.json after every layer (layer, seconds per layer,
ETA). Usage (PTQ image, CPU): bf16_teacher_cpu.py SOURCE_DIR HELDOUT_JSONL OUT_JSON WORK_DIR [THREADS]
Prints TEACHER-KNOWN-ANSWER-OK, then TEACHER-OK when OUT_JSON is written."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

MODELING_SHA256 = "f83865ef643e273a6c9aecf1575e85c95ad01fd995b1e43c3c7a892765006bb3"
CONFIGURATION_SHA256 = "002cb3915be9685eb962d617623bec14d45b5d6b5d46e689fac98a3b38cb6cf6"
KNOWN = [("The capital of France is", " Paris"), ("The chemical symbol for gold is", " Au")]
TOP_K = 20
SAVE_EVERY = 4
LM_CHUNK = 512


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


class Streamer:
    def __init__(self, src: Path):
        from transformers import AutoConfig, AutoModelForCausalLM

        for name, want in (("modeling_ricursive.py", MODELING_SHA256), ("configuration_ricursive.py",
                                                                        CONFIGURATION_SHA256)):
            if sha(src / name) != want:
                raise SystemExit(f"TEACHER-FAILED: {name} is not the pinned remote code")
        self.src = src
        self.cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        with torch.device("meta"):
            shell = AutoModelForCausalLM.from_config(self.cfg, trust_remote_code=True, dtype=torch.float32)
        self.mod = sys.modules[type(shell).__module__]
        del shell
        self.idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles: dict = {}

    def get(self, key: str) -> torch.Tensor:
        f = self.idx[key]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.src / f), "pt")
        return self.handles[f].get_tensor(key).float()

    def layer(self, i: int):
        with torch.device("meta"):
            layer = self.mod.RicursiveDecoderLayer(self.cfg, i)
        pre = f"model.layers.{i}."
        sd = {k[len(pre):]: self.get(k) for k in self.idx if k.startswith(pre)}
        layer.load_state_dict(sd, strict=True, assign=True)
        return layer.eval()

    def embed(self, seqs: list[torch.Tensor]) -> list[torch.Tensor]:
        emb = self.get("model.embed_tokens.weight")
        scale = getattr(self.cfg, "embed_scale", None) or 1.0
        out = [emb[s][None] * scale for s in seqs]
        del emb
        return out

    @staticmethod
    def run_layer(layer, h: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        t = h.shape[1]
        return layer(h, attention_mask=None, position_ids=torch.arange(t)[None], past_key_value=None,
                     use_cache=False, cache_position=torch.arange(t), value_residual=vr)[0]

    def head(self):
        return self.get("model.norm.weight"), self.get("lm_head.weight")

    def norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps))


def known_answer_check(s: Streamer, tok) -> dict:
    seqs = [torch.tensor(tok(p, add_special_tokens=False)["input_ids"]) for p, _ in KNOWN]
    hs = s.embed(seqs)
    vr = [h.clone() for h in hs]
    with torch.no_grad():
        for i in range(s.cfg.num_hidden_layers):
            layer = s.layer(i)
            hs = [s.run_layer(layer, h, v) for h, v in zip(hs, vr)]
            del layer
        nw, lm = s.head()
        got = {}
        for (p, want), h in zip(KNOWN, hs):
            lp = torch.log_softmax(s.norm(h[0, -1], nw) @ lm.T, -1)
            top = torch.topk(lp, 3)
            got[p] = {"want": want, "top": [(tok.decode([int(t)]), round(float(v), 4)) for v, t in
                                             zip(top.values, top.indices)]}
            if tok.decode([int(top.indices[0])]) != want:
                raise SystemExit(f"TEACHER-FAILED: known answer for {p!r} is {got[p]['top'][0]}, not {want!r}")
    return got


def main() -> None:
    src, held_path, out_path, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    threads = int(sys.argv[5]) if len(sys.argv) > 5 else 32
    torch.set_num_threads(threads)
    work.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer

    import transformers

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(src)
    s = Streamer(src)
    held_bytes = held_path.read_bytes()
    held_sha = hashlib.sha256(held_bytes).hexdigest()
    prompts = [json.loads(line) for line in held_bytes.split(b"\n") if line.strip()]
    if len(prompts) != 64:
        raise SystemExit(f"TEACHER-FAILED: {len(prompts)} held-out prompts, expected 64")
    for p in prompts:
        if tok(p["text"], add_special_tokens=False)["input_ids"] != p["token_ids"]:
            raise SystemExit(f"TEACHER-FAILED: token ids of {p['file']}:{p['index']} differ from its text")
    progress = work / "progress.json"

    def write_progress(d: dict) -> None:
        tmp = progress.with_suffix(".tmp")
        tmp.write_text(json.dumps(d) + "\n")
        tmp.rename(progress)

    ka_path = work / "known_answer.json"
    if not ka_path.exists():
        write_progress({"stage": "known-answer", "started": time.strftime("%FT%TZ", time.gmtime())})
        ka = known_answer_check(s, tok)
        ka_path.write_text(json.dumps(ka) + "\n")
    ka = json.loads(ka_path.read_text())
    print(f"TEACHER-KNOWN-ANSWER-OK transformers {transformers.__version__} torch {torch.__version__} "
          f"{json.dumps(ka)} ({time.time() - t0:.0f}s)", flush=True)

    n_layers = s.cfg.num_hidden_layers
    seqs = [torch.tensor(p["token_ids"]) for p in prompts]
    total_tokens = sum(len(x) for x in seqs)
    state_path = work / "state.pt"
    start = 0
    hs = None
    if state_path.exists():
        st = torch.load(state_path, map_location="cpu", weights_only=True)
        if st.get("heldout_sha256") == held_sha:
            start, hs = int(st["next_layer"]), list(st["hidden"])
            print(f"resuming at layer {start}/{n_layers}", flush=True)
    vr = s.embed(seqs)
    if hs is None:
        hs = [v.clone() for v in vr]
    per_layer: list[float] = []
    with torch.no_grad():
        for i in range(start, n_layers):
            t1 = time.time()
            layer = s.layer(i)
            hs = [s.run_layer(layer, h, v) for h, v in zip(hs, vr)]
            del layer
            if not all(bool(torch.isfinite(h).all()) for h in hs):
                raise SystemExit(f"TEACHER-FAILED: non-finite hidden state after layer {i}")
            per_layer.append(time.time() - t1)
            if (i + 1) % SAVE_EVERY == 0 and i + 1 < n_layers:
                tmp = work / "state.tmp"
                torch.save({"heldout_sha256": held_sha, "next_layer": i + 1, "hidden": hs}, tmp)
                tmp.rename(state_path)
            rate = sum(per_layer[-8:]) / len(per_layer[-8:])
            write_progress({"stage": "layers", "layer_done": i + 1, "of": n_layers, "s_last_layer": round(per_layer[-1], 1),
                            "s_per_layer_recent": round(rate, 1), "eta_s": round(rate * (n_layers - i - 1) + 300),
                            "prompts": len(seqs), "tokens": total_tokens, "elapsed_s": round(time.time() - t0)})
            print(f"layer {i + 1}/{n_layers} {per_layer[-1]:.1f}s (recent {rate:.1f}s/layer, ETA "
                  f"{rate * (n_layers - i - 1) / 3600:.2f} h)", flush=True)
        write_progress({"stage": "head", "layer_done": n_layers, "of": n_layers, "elapsed_s": round(time.time() - t0)})
        nw, lm = s.head()
        rows = []
        top1 = []
        for p, h, ids in zip(prompts, hs, seqs):
            x = s.norm(h[0], nw)
            pos = []
            for a in range(0, x.shape[0] - 1, LM_CHUNK):
                b = min(a + LM_CHUNK, x.shape[0] - 1)
                lp = torch.log_softmax(x[a:b] @ lm.T, -1)
                top = torch.topk(lp, TOP_K, dim=-1)
                nxt = ids[a + 1:b + 1]
                act = lp.gather(1, nxt[:, None])[:, 0]
                for r in range(b - a):
                    d = {str(int(t)): float(v) for v, t in zip(top.values[r], top.indices[r])}
                    d.setdefault(str(int(nxt[r])), float(act[r]))
                    pos.append(d)
                    top1.append(int(top.indices[r, 0]) == int(nxt[r]))
            if len(pos) != len(ids) - 1:
                raise SystemExit("TEACHER-FAILED: position count")
            rows.append(pos)
    out = {"model": "ripple-65b-bf16-cpu-teacher-fp32", "heldout_sha256": held_sha,
           "transformers": transformers.__version__, "torch": torch.__version__, "known_answer": ka,
           "prompts": [{k: p[k] for k in ("source", "file", "index", "tokens", "span_start")} for p in prompts],
           "rows": rows}
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(json.dumps(out))
    tmp.rename(out_path)
    if state_path.exists():
        state_path.unlink()
    write_progress({"stage": "done", "layer_done": n_layers, "of": n_layers, "elapsed_s": round(time.time() - t0),
                    "out_sha256": sha(out_path)})
    print(f"TEACHER-OK {out_path} sha256 {sha(out_path)} | {len(rows)} prompts, {sum(len(r) for r in rows)} positions, "
          f"teacher top-1 = next token on {sum(top1) / len(top1):.4f} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
