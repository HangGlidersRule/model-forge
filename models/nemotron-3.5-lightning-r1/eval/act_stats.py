#!/usr/bin/env python3
"""Issue #17 R17-A1: activation statistics for candidate A's 92 activation quantizers, from the BF16 model.

Per MoE layer, three input streams (routed and shared up_proj read the same tensor, the MoE input):
  L<i>.up     MoE input hidden states            -> routed experts up_proj + shared expert up_proj
  L<i>.rdown  relu2 outputs of the routed experts -> routed experts down_proj (one scale shared by all 128)
  L<i>.sdown  relu2 output of the shared expert   -> shared expert down_proj
For each stream: exact max |x|, a histogram of per-block (16) amax in 1/8-octave bins, and the simulated NVFP4
squared error for a grid of global amax values G (half-octave steps, 2^-4 .. 2^16). Simulation follows the NVFP4
W4A4 activation path: global scale gs = 6*448/G, per-block scale e4m3(amax_block/6*gs) saturating at 448, values on the
e2m1 grid {0,.5,1,1.5,2,3,4,6} clipped at 6.
Usage: act_stats.py BF16_DIR CALIB_JSONL OUT_JSON N_SAMPLES"""
import json
import sys
import time

import torch
import transformers.integrations.moe as tmoe
from transformers import AutoModelForCausalLM, AutoTokenizer

src, calib, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tmoe._can_use_grouped_mm = lambda *a, **k: False  # SM120: the grouped_mm capability gate is wrong; force eager experts
LO, HI, NB = -30.0, 20.0, 400
GRID = [2.0 ** (j / 2) for j in range(-8, 33)]  # 2^-4 .. 2^16
KC = 8
dev = "cuda"
G = torch.tensor(GRID, device=dev, dtype=torch.float32)
GS = 2688.0 / G
S = {}


def acc(name, x):
    xb = x.reshape(-1, x.shape[-1]).float().reshape(-1, 16)
    bam = xb.abs().amax(-1)
    st = S.get(name)
    if st is None:
        st = S[name] = {"max": 0.0, "n": 0, "zero": 0, "ss": 0.0,
                        "hist": torch.zeros(NB, dtype=torch.float64, device=dev),
                        "sse": torch.zeros(len(GRID), dtype=torch.float64, device=dev)}
    st["max"] = max(st["max"], float(bam.max()))
    nz = bam > 0
    st["n"] += bam.numel()
    st["zero"] += int((~nz).sum())
    st["hist"] += torch.histc(torch.log2(bam[nz]).clamp(LO, HI - 1e-6), bins=NB, min=LO, max=HI).double()
    st["ss"] += float((xb * xb).sum())
    for k0 in range(0, len(GRID), KC):
        gs = GS[k0:k0 + KC]
        sq = (bam[None] / 6.0 * gs[:, None]).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        deq = sq / gs[:, None]
        y = xb[None] / deq[..., None]
        a = y.abs().clamp(max=6.0)
        q = torch.where(a <= 2, torch.round(a * 2) / 2, torch.where(a <= 4, torch.round(a), torch.round(a / 2) * 2))
        xh = torch.where(deq[..., None] > 0, torch.sign(y) * q * deq[..., None], torch.zeros_like(y))
        st["sse"][k0:k0 + KC] += ((xb[None] - xh) ** 2).sum(dim=(1, 2)).double()


tok = AutoTokenizer.from_pretrained(src)
model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16, device_map=dev)
model.eval()
for m in model.modules():
    cfg = getattr(m, "config", None)
    if cfg is not None and hasattr(cfg, "_experts_implementation"):
        cfg._experts_implementation = "eager"
buf = {}
hooks = 0
for i, layer in enumerate(model.model.layers):
    mx = layer.mixer
    if type(mx).__name__ != "NemotronHMoE":
        continue
    assert mx.experts.act_fn is not mx.shared_experts.act_fn
    mx.register_forward_pre_hook(lambda mod, args, nm=f"L{i}.up": acc(nm, args[0]))
    mx.experts.act_fn.register_forward_hook(lambda mod, inp, o, nm=f"L{i}.rdown": buf.setdefault(nm, []).append(o.detach()))
    mx.experts.register_forward_hook(lambda mod, inp, o, nm=f"L{i}.rdown": acc(nm, torch.cat(buf.pop(nm), 0)) if buf.get(nm) else None)
    mx.shared_experts.down_proj.register_forward_pre_hook(lambda mod, args, nm=f"L{i}.sdown": acc(nm, args[0]))
    hooks += 1
print(f"hooked {hooks} MoE layers", flush=True)
rows = [json.loads(line)["text"] for line in open(calib, encoding="utf-8")][:n]
t0, ntok = time.time(), 0
with torch.no_grad():
    for k, text in enumerate(rows):
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).input_ids.to(dev)
        model(input_ids=ids, use_cache=False)
        ntok += ids.shape[1]
        if k % 16 == 0:
            print(f"{k + 1}/{len(rows)} samples, {ntok} tokens, {time.time() - t0:.0f} s", flush=True)
res = {"samples": len(rows), "tokens": ntok, "grid": GRID, "hist_lo": LO, "hist_hi": HI, "hist_bins": NB,
       "streams": {k: {"max": v["max"], "n": v["n"], "zero": v["zero"], "ss": v["ss"],
                       "hist": v["hist"].tolist(), "sse": v["sse"].tolist()} for k, v in sorted(S.items())}}
json.dump(res, open(out, "w"))
print(json.dumps({"streams": len(S), "samples": len(rows), "tokens": ntok, "seconds": round(time.time() - t0)}))
