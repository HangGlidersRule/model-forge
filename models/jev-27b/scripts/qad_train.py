#!/usr/bin/env python3
"""Issue #20 QAD trainer, run 1 (pre-registration qad-run1-preregistration.md, sha256 38542083...).

Learns the NVFP4 per-block weight scales of the all-W4A4 JEV-27B (layout B) against the corpus teacher.
  Start point: the merged BF16 weights with candidate B's exported scales loaded exactly (per-block weight scales,
    per-tensor weight scales, per-tensor activation scales), so step 0 reproduces B's served numerics. Checked at
    startup against B's packed FP4 weights.
  Trained: only the per-block weight amax (ModelOpt LSQ, tied pre/post, fp32 master values, FP8-quantized in the
    forward exactly as exported). Frozen: BF16 weights, per-tensor scales, activation quantization, lm_head,
    embeddings, norms, conv1d, in_proj_a/b, vision tower.
  Objective: KL(target || readout). The readout is serve_decide.py's System 1 distribution: verbalizer logits after
    "[decision]:" plus the decision-head bias, divided by the per-kind temperature, softmax over the row's options.
  Data: jev-distill-corpus-v3 train (seeded, stratified noul 41 / choice 31 / score 28; 3-4-option choice rows 2x),
    2,000-row validation subset for checkpoint selection. test_set_30k is never read.
  Schedule: fused AdamW lr 1e-4 on the amax only, wd 0; 50-step warmup, cosine to 10%; ~16k prompt tokens per step;
    <= 1,500 steps or 6 h; validate every 100 steps; keep the best; stop after 4 evaluations without improvement.
  Export: fold the learned amax back into static quantizers, export with ModelOpt's unified HF exporter (as hf_ptq
    does for this VLM), then check the export against the trained fake-quant and against B's tensor layout.

Stack (image local/model-forge-qad:20261028-t2141): torch 2.14.1+cu132 + triton 3.8.0 (torch 2.15/triton 3.9 removed the
block-pointer API that ModelOpt main's NVFP4 activation kernel uses), ModelOpt main b6b0ff5, transformers 5.18.0,
Gated DeltaNet on FLA 0.6.0 + causal-conv1d 1.7.0, full attention on PyTorch SDPA's flash kernel (FlashAttention-4's
sm_120 kernels cannot fit head_dim 256 in the 99 KB of shared memory per SM).

Usage:
  qad_train.py --model DIR --init-export DIR --recipe YAML --train JSONL --val JSONL --head DIR --out DIR
               [--max-steps 1500] [--max-hours 6] [--tokens-per-step 16384] [--micro-tokens 8192] [--dry-run N]
--dry-run N: N optimizer steps, timing/memory report, then fold + export + export checks (no checkpoint selection).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

LABELS = [chr(ord("A") + i) for i in range(26)]
SEED = 1234
E2M1_MAX = 6.0
E4M3_MAX = 448.0
N_NVFP4_LINEARS = 400  # 192 MLP + 64 full-attention + 144 DeltaNet projections (layout B)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------- prompt + readout
def render(r: dict) -> str:
    """serve_decide.py's text-only System 1 prompt (Qwen profile: empty prefix, no chat template)."""
    kind, opts = r["kind"], list(r.get("options") or [])
    if kind == "choice":
        lines = [f"{LABELS[i]}) {o}" for i, o in enumerate(opts)]
    elif kind == "noul":
        lines = ["false", "true"]
    else:
        lines = [str(i) for i in range(6)]
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    return f"[kind] {kind}\n[state] {st}\n[question] {r['question']}\n[options]\n" + "\n".join(lines) + "\n[decision]:"


class Readout:
    """serve_decide.py's System 1 readout: verbalizer ids + bias per kind/option, divided by the per-kind temperature."""

    def __init__(self, head_dir: Path, tok):
        h = json.loads((head_dir / "decision_head.json").read_text())
        cal = json.loads((head_dir / "calibration.json").read_text())["per_kind"]
        self.ranges, self.vids, self.bias, self.temps = h["slots"]["ranges"], h["verbalizer_ids"], h["bias"], cal
        lo, hi = self.ranges["choice"]
        self.choice_ids = self.vids[lo:hi]
        # serve_decide reads choice labels A..P and asserts they are the trained head ids
        enc = [tok.encode(c, add_special_tokens=False) for c in LABELS[: hi - lo]]
        assert [e[0] for e in enc] == self.choice_ids, "choice label ids differ from the trained head"

    def ids_bias(self, kind: str, n_opts: int):
        lo, hi = self.ranges[kind]
        if kind == "choice":
            return self.choice_ids[:n_opts], [self.bias[lo + i] if lo + i < hi else 0.0 for i in range(n_opts)]
        return self.vids[lo:hi], self.bias[lo:hi]

    def logprobs(self, last_logits: torch.Tensor, rows: list[dict]) -> list[torch.Tensor]:
        out = []
        for i, r in enumerate(rows):
            ids, b = self.ids_bias(r["kind"], len(r["options"]) if r["kind"] == "choice" else 0)
            z = (last_logits[i, ids].float() + torch.tensor(b, device=last_logits.device)) / self.temps[r["kind"]]
            out.append(F.log_softmax(z, -1))
        return out


# ----------------------------------------------------------------------------------------------------------- data
def load_rows(path: Path) -> list[dict]:
    """JSONL rows split on b"\\n" only. str.splitlines() also splits on U+2028/U+2029/U+0085, which JSON allows
    unescaped inside strings (the corpus has them), and would cut rows in half."""
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line.strip()]


def weighted_sample(pool: list, weights: list[float], k: int, rng: random.Random) -> list:
    """Seeded weighted sampling without replacement (Efraimidis-Spirakis keys)."""
    keys = sorted(((rng.random() ** (1.0 / w), i) for i, w in enumerate(weights)), reverse=True)
    return [pool[i] for _, i in keys[:k]]


def stratified(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Seeded draw matching the evaluation mix (noul 41 / choice 31 / score 28); 3-4-option choice rows weighted 2x.
    Choice rows with more than 16 options (multi-pass at serve time) are not used."""
    mix = {"noul": 0.41, "choice": 0.31, "score": 0.28}
    out = []
    for kind, share in mix.items():
        pool = [r for r in rows if r["kind"] == kind and (kind != "choice" or len(r["options"]) <= 16)]
        w = [2.0 if kind == "choice" and 3 <= len(r["options"]) <= 4 else 1.0 for r in pool]
        out += weighted_sample(pool, w, min(len(pool), round(n * share)), rng)
    rng.shuffle(out)
    return out


def encode(rows: list[dict], tok, max_len: int = 2048) -> list[tuple[list[int], dict]]:
    ids = tok([render(r) for r in rows], add_special_tokens=False)["input_ids"]
    return [(i, r) for i, r in zip(ids, rows) if len(i) <= max_len]


def make_steps(enc: list, tokens_per_step: int, rng: random.Random) -> list[list]:
    """Seeded shuffle, then consecutive groups of ~tokens_per_step prompt tokens (one optimizer step each)."""
    order = list(range(len(enc)))
    rng.shuffle(order)
    steps, cur, toks = [], [], 0
    for i in order:
        cur.append(enc[i])
        toks += len(enc[i][0])
        if toks >= tokens_per_step:
            steps.append(cur)
            cur, toks = [], 0
    if cur:
        steps.append(cur)
    return steps


def micro_batches(rows: list, micro_tokens: int) -> list[list]:
    """Length-sorted micro-batches of at most micro_tokens padded tokens."""
    out, cur, cur_max = [], [], 0
    for ids, r in sorted(rows, key=lambda x: len(x[0])):
        m = max(cur_max, len(ids))
        if cur and m * (len(cur) + 1) > micro_tokens:
            out.append(cur)
            cur, m = [], len(ids)
        cur.append((ids, r))
        cur_max = m
    if cur:
        out.append(cur)
    return out


def collate(batch, pad_id: int, device):
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L), pad_id, dtype=torch.long)
    last = torch.zeros(len(batch), dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, : len(ids)] = torch.tensor(ids)
        last[i] = len(ids) - 1
    return x.to(device), last.to(device), [r for _, r in batch]


def text_model(model):
    """The decoder stack a text-only System 1 prompt runs through (Qwen3.5 VLM: model.model.language_model)."""
    inner = model.model
    return getattr(inner, "language_model", inner)


def batch_kl(model, readout: Readout, batch, pad_id: int, device) -> tuple[torch.Tensor, int]:
    """Summed KL(target || readout) over a right-padded micro-batch. Attention, the DeltaNet recurrence and its short
    conv are all causal, so real positions never see the pads that follow them: no mask is passed (SDPA stays on its
    flash kernel) and only the last real token of each row is read. Activation scales are static (B's), so pads do not
    influence quantization either."""
    x, last, rows = collate(batch, pad_id, device)
    hidden = text_model(model)(input_ids=x, use_cache=False).last_hidden_state
    h_last = hidden[torch.arange(len(rows), device=device), last]
    lps = readout.logprobs(model.lm_head(h_last), rows)
    kl = torch.zeros((), device=device, dtype=torch.float32)
    for lp, r in zip(lps, rows):
        t = torch.tensor(r["target"], device=device, dtype=torch.float32)
        kl = kl + (t * (t.clamp_min(1e-12).log() - lp)).sum()
    return kl, len(rows)


def enabled_quantizers(model) -> list:
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

    return [m for m in model.modules() if isinstance(m, TensorQuantizer) and m.is_enabled]


class TeacherMode:
    """Run the model as its own BF16 teacher: every enabled quantizer is bypassed inside the block (a disabled
    TensorQuantizer passes its input through unchanged), so the teacher needs no second copy of the weights."""

    def __init__(self, qs: list):
        self.qs = qs

    def __enter__(self):
        for q in self.qs:
            q.disable()

    def __exit__(self, *exc):
        for q in self.qs:
            q.enable()


def distill_kl(model, batch, pad_id: int, device, qs: list, chunk: int = 512) -> tuple[torch.Tensor, int]:
    """Summed full-vocabulary KL(BF16 teacher || NVFP4 student) over every real position of a right-padded
    micro-batch: NVIDIA QAD's logits loss (LogitsDistillationLoss, temperature 1). Logits are formed in position
    chunks under activation checkpointing, so the [tokens x 248k] tensors never all exist at once."""
    x, last, _ = collate(batch, pad_id, device)
    mask = torch.arange(x.shape[1], device=device)[None, :] <= last[:, None]
    tm, head = text_model(model), model.lm_head
    with torch.no_grad(), TeacherMode(qs):
        ht = tm(input_ids=x, use_cache=False).last_hidden_state[mask]
    hs = tm(input_ids=x, use_cache=False).last_hidden_state[mask]

    def chunk_kl(a, b):
        ls = F.log_softmax(head(a).float(), -1)
        with torch.no_grad():
            lt = F.log_softmax(head(b).float(), -1)
        return F.kl_div(ls, lt, log_target=True, reduction="sum")

    total = torch.zeros((), device=device, dtype=torch.float32)
    for i in range(0, hs.shape[0], chunk):
        if torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(chunk_kl, hs[i:i + chunk], ht[i:i + chunk],
                                                              use_reentrant=False)
        else:
            total = total + chunk_kl(hs[i:i + chunk], ht[i:i + chunk])
    return total, int(mask.sum())


# -------------------------------------------------------------------------------------------- model + quantization
def load_student(model_dir: Path, auto_class=None):
    """Load through ModelOpt's loader (records checkpoint tensors the model has no parameter for, e.g. the MTP head,
    so the exporter carries them over), with the architecture class from config.json as hf_ptq does."""
    import transformers
    from modelopt.torch.utils.plugins.model_load_utils import load_pretrained

    if auto_class is None:
        arch = json.loads((model_dir / "config.json").read_text())["architectures"][0]
        auto_class = getattr(transformers, arch)
    model = load_pretrained(str(model_dir), auto_class, dtype=torch.bfloat16, attn_implementation="sdpa",
                            device_map={"": 0})
    model.config.use_cache = False
    return model


def quantize_structure(model, recipe_path: Path, rename: dict | None = None, forward_loop=None):
    """Insert the layout-B quantizers (the recipe's quant_cfg) with plain max calibration; the scales are replaced by
    load_scales_from_export, so only the structure matters here."""
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_recipe

    cfg = load_recipe(str(recipe_path)).quantize.model_dump()
    if rename:
        for e in cfg["quant_cfg"]:
            if isinstance(e, dict) and isinstance(e.get("quantizer_name"), str):
                for old, new in rename.items():
                    e["quantizer_name"] = e["quantizer_name"].replace(old, new)
    cfg["algorithm"] = {"method": "max"}
    if forward_loop is None:
        dev = next(model.parameters()).device

        def forward_loop(m):
            with torch.no_grad():
                m(input_ids=torch.zeros((1, 16), dtype=torch.long, device=dev))
    model = mtq.quantize(model, cfg, forward_loop)
    # Static NVFP4 weight quantizers must be StaticBlockScaleQuantizer (per-block _amax + global_amax) for LSQ and for
    # the static export path; promotion is idempotent.
    from modelopt.torch.quantization.utils.core_utils import promote_static_block_weight_quantizers

    promote_static_block_weight_quantizers(model)
    return model


def nvfp4_linears(model) -> list[tuple[str, torch.nn.Module]]:
    out = []
    for name, mod in model.named_modules():
        wq = getattr(mod, "weight_quantizer", None)
        if wq is not None and getattr(wq, "is_enabled", False) and isinstance(getattr(mod, "weight", None), torch.Tensor):
            out.append((name, mod))
    return out


class ExportReader:
    """Random access to the tensors of a safetensors checkpoint directory."""

    def __init__(self, d: Path):
        from safetensors import safe_open

        self.d, self._h = Path(d), {}
        idx = self.d / "model.safetensors.index.json"
        if idx.exists():
            self.map = json.loads(idx.read_text())["weight_map"]
        else:
            self.map = {}
            for f in sorted(self.d.glob("*.safetensors")):
                with safe_open(str(f), "pt") as h:
                    for k in h.keys():
                        self.map[k] = f.name

    def keys(self):
        return self.map.keys()

    def get(self, k: str) -> torch.Tensor:
        from safetensors import safe_open

        f = self.map[k]
        if f not in self._h:
            self._h[f] = safe_open(str(self.d / f), "pt", device="cpu")
        return self._h[f].get_tensor(k)


def exact_preimage(target: torch.Tensor, div: float) -> torch.Tensor:
    """Float32 g with g / div == target exactly in float32. ModelOpt exports per-tensor scales as amax.float() / div,
    so an amax rebuilt with this writes the reference checkpoint's scale back byte-for-byte (a plain target * div can be
    one float32 ulp off). Stops at the closest candidate if no exact preimage exists."""
    t = target.detach().float().cpu().reshape(-1)
    g = (t.double() * div).float()
    up, down = torch.full_like(g, float("inf")), torch.full_like(g, float("-inf"))
    for _ in range(16):
        back = g / div
        lo, hi = back < t, back > t
        if not bool((lo | hi).any()):
            break
        g = torch.where(lo, torch.nextafter(g, up), torch.where(hi, torch.nextafter(g, down), g))
    return g.reshape(target.shape)


def load_scales_from_export(model, ref: ExportReader) -> int:
    """Set every NVFP4 linear's quantizers to the exported checkpoint's exact values:
       weight_scale_2 = global_amax / (6 * 448)          -> weight_quantizer.global_amax
       weight_scale   = E4M3(amax * 448 / global_amax)   -> weight_quantizer._amax = weight_scale * weight_scale_2 * 6
       input_scale    = activation amax / (6 * 448)      -> input_quantizer.amax
    The reconstructed per-block amax sits exactly on the E4M3 grid, so the forward re-quantizes it to the same scale."""
    n = 0
    for name, mod in nvfp4_linears(model):
        wq, iq, dev = mod.weight_quantizer, mod.input_quantizer, mod.weight.device
        bs = ref.get(f"{name}.weight_scale").float()
        ws2 = ref.get(f"{name}.weight_scale_2").float()
        assert tuple(bs.shape) == (mod.weight.shape[0], mod.weight.shape[1] // 16), (name, tuple(bs.shape))
        wq.global_amax = exact_preimage(ws2, E2M1_MAX * E4M3_MAX).to(dev)
        amax = (bs * ws2 * E2M1_MAX).to(dev)
        assert wq._amax.numel() == amax.numel(), (name, wq._amax.shape, amax.shape)
        wq._amax.data.copy_(amax.reshape(wq._amax.shape).to(wq._amax.dtype))
        iq.amax = exact_preimage(ref.get(f"{name}.input_scale").float(), E2M1_MAX * E4M3_MAX).to(dev)
        n += 1
    for name, mod in nvfp4_linears(model):  # fused groups share global_amax: every member must still match
        want = float(ref.get(f"{name}.weight_scale_2").float()) * E2M1_MAX * E4M3_MAX
        got = float(mod.weight_quantizer.global_amax)
        assert abs(got - want) <= 1e-6 * abs(want), (name, got, want)
    return n


def sample_linears(names: list[str]) -> list[str]:
    """First and last module of every projection type (20 linears for layout B)."""
    by: dict[str, list[str]] = {}
    for n in names:
        by.setdefault(".".join(n.split(".")[-2:]), []).append(n)
    return sorted({v[0] for v in by.values()} | {v[-1] for v in by.values()})


def fake_quant_weights(model, names: list[str]) -> dict[str, torch.Tensor]:
    """CPU copies of the model's current fake-quantized weights. Take them before exporting: ModelOpt's export packs
    the module weights in place, after which the quantizers cannot be applied to them."""
    out = {}
    with torch.no_grad():
        for name in names:
            mod = model.get_submodule(name)
            out[name] = mod.weight_quantizer(mod.weight).detach().to("cpu", copy=True)
    return out


def compare_to_export(fq: dict[str, torch.Tensor], ref: ExportReader, tol: float = 2.0 ** -7) -> dict:
    """Fake-quantized weights vs the dequantized packed weights of a checkpoint. A different FP4 code moves an element
    by >= 33%, a different E4M3 block scale by >= 6%; bf16 output rounding stays below 2^-8. A block with >= 8 of its 16
    elements mismatched points at its scale; isolated elements point at rounding near an FP4 boundary."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    dev = torch.device("cuda")
    bad = blocks_any = blocks_scale = total = 0
    per = []
    for name, w in fq.items():
        w = w.to(dev).float()
        deq = NVFP4QTensor(w.shape, torch.float32, ref.get(f"{name}.weight").to(dev)).dequantize(
            dtype=torch.float32, scale=ref.get(f"{name}.weight_scale").to(dev),
            double_scale=ref.get(f"{name}.weight_scale_2").to(dev), block_sizes={-1: 16})
        rel = (deq - w).abs() / torch.maximum(deq.abs(), w.abs()).clamp_min(1e-30)
        m = (rel > tol).reshape(-1, 16)
        per_block = m.sum(-1)
        b = int(m.sum())
        bad += b
        total += w.numel()
        blocks_any += int((per_block > 0).sum())
        blocks_scale += int((per_block >= 8).sum())
        if b:
            per.append((b, name))
        del w, deq, rel, m, per_block
    per.sort(reverse=True)
    return {"elements": total, "mismatched": bad, "fraction": bad / max(1, total), "blocks_with_mismatch": blocks_any,
            "blocks_scale_like": blocks_scale, "worst": [f"{n}:{b}" for b, n in per[:5]]}


def export_matches(chk: dict, max_fraction: float = 1e-3) -> bool:
    """Pass rule for a fake-quant vs export comparison: no block-scale-level mismatches, and isolated element flips
    (FP4 rounding at a code boundary, from the same float32 math done in a different order) below 0.1% of elements.
    A real bug (wrong scale, layout or rounding rule) moves whole blocks or far more than 0.1% of elements."""
    return chk["blocks_scale_like"] == 0 and chk["fraction"] <= max_fraction


class Movement:
    """Proof that training changes what the model computes, not just the master copy of the scales.
    The forward uses scale = E4M3(amax / 6 under the frozen per-tensor scale), so an amax update only matters once it
    crosses an E4M3 rounding boundary. Tracks, against the step-0 values:
      moved_amax:  share of blocks whose learned amax changed at all
      moved_scale: share of blocks whose FP8 block scale (what the forward and the export use) changed
      rel_change:  median and p99 of |amax - amax0| / amax0
    plus the share of blocks that received a nonzero gradient in the last step."""

    def __init__(self, qmods: list, params: list):
        self.qmods, self.params = qmods, params
        self.a0 = [p.detach().float().cpu().clone() for p in params]
        self.s0 = [self._scale(q).cpu() for q in qmods]

    @staticmethod
    def _scale(q) -> torch.Tensor:
        with torch.no_grad():
            return q._block_scale_from_amax(q.amax_post, q._quantize_scales).detach().float().reshape(-1)

    def report(self, grads: bool = True) -> dict:
        n = moved_a = moved_s = gnz = 0
        rel = []
        for p, a0, q, s0 in zip(self.params, self.a0, self.qmods, self.s0):
            a = p.detach().float().cpu()
            moved_a += int((a != a0).sum())
            moved_s += int((self._scale(q).cpu() != s0).sum())
            if grads and p.grad is not None:
                gnz += int((p.grad != 0).sum())
            n += a.numel()
            r = ((a - a0).abs() / a0.abs().clamp_min(1e-30)).reshape(-1)
            rel.append(r[torch.randint(0, r.numel(), (min(4096, r.numel()),))])
        r = torch.cat(rel).sort().values
        return {"moved_amax": moved_a / n, "moved_scale": moved_s / n, "grad_nonzero": gnz / n if grads else None,
                "rel_change_p50": float(r[len(r) // 2]), "rel_change_p99": float(r[int(len(r) * 0.99)])}


def enable_lsq(model, dtype=torch.float32) -> list[torch.nn.Parameter]:
    """ModelOpt LSQ on every NVFP4 weight quantizer: tied pre/post amax, FP8-quantized scales in the forward.
    dtype is the learnable amax dtype: ModelOpt's own lsq() uses the weight dtype (bf16); run 1 used fp32.
    Everything else is frozen."""
    from modelopt.torch.quantization.nn.modules.tensor_quantizer import StaticBlockScaleQuantizer

    for p in model.parameters():
        p.requires_grad_(False)
    params = []
    for name, mod in nvfp4_linears(model):
        wq = mod.weight_quantizer
        assert isinstance(wq, StaticBlockScaleQuantizer) and hasattr(wq, "_amax"), name
        wq.enable_lsq(learnable_amax=["pre", "post"], tied_amax=True, quantize_pre_scale=True, dtype=dtype)
        assert wq._amax_post.requires_grad
        params.append(wq._amax_post)
    return params


def fold_lsq(model) -> int:
    """Write each learned tied amax back as the static _amax and leave LSQ mode: ModelOpt's NVFP4 export reads
    weight_quantizer._amax, and the static fake-quant of the folded amax equals the trained LSQ forward."""
    n = 0
    with torch.no_grad():
        for _, mod in nvfp4_linears(model):
            wq = mod.weight_quantizer
            if not getattr(wq, "_lsq", False):
                continue
            learned = wq._amax_post.detach().float().clone()
            for attr in ("_amax_post", "_amax_pre"):
                if attr in wq._parameters or attr in wq._buffers:
                    delattr(wq, attr)
            wq.register_buffer("_amax", learned)
            wq._lsq, wq._learnable_amax, wq._tied_amax = False, [], False
            n += 1
    return n


def export(model, src_dir: Path, out_dir: Path) -> None:
    """Unified HF export, as hf_ptq does it: VLMs first get the source config (the exporter skips it)."""
    from modelopt.torch.export import export_hf_checkpoint
    from modelopt.torch.export.model_utils import is_multimodal_model
    from transformers import AutoConfig

    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        if is_multimodal_model(model):
            AutoConfig.from_pretrained(str(src_dir)).save_pretrained(str(out_dir))
        export_hf_checkpoint(model, export_dir=str(out_dir))


def compare_exports(new: ExportReader, ref: ExportReader, trained_linears: list[str]) -> dict:
    """Expected: exactly the reference's tensors, dtypes and shapes; every tensor other than the trained NVFP4 weights
    and their per-block scales byte-identical (BF16 weights, norms, conv, embeddings, vision, MTP, per-tensor weight
    and activation scales); hf_quant_config.json equal except the producer. Returns a report with an "issues" list
    (empty = pass) instead of raising, so a metadata difference after a long run is reviewed rather than fatal."""
    issues = []
    nk, rk = set(new.keys()), set(ref.keys())
    if nk != rk:
        issues.append({"tensor_sets_differ": {"only_new": sorted(nk - rk)[:10], "only_ref": sorted(rk - nk)[:10],
                                              "n_only_new": len(nk - rk), "n_only_ref": len(rk - nk)}})
    trained = {f"{n}.{s}" for n in trained_linears for s in ("weight", "weight_scale")}
    changed_trained, n_identical, untrained_changed, ulp_drift = 0, 0, [], []
    for k in sorted(nk & rk):
        a, b = new.get(k), ref.get(k)
        if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
            issues.append({"dtype_or_shape": [k, str(a.dtype), str(b.dtype), list(a.shape), list(b.shape)]})
            continue
        same = torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))
        if k in trained:
            changed_trained += int(not same)
        elif same:
            n_identical += 1
        elif a.numel() == 1 and a.dtype == torch.float32 and abs(
                int(a.reshape(1).view(torch.int32)) - int(b.reshape(1).view(torch.int32))) <= 2:
            ulp_drift.append(k)  # per-tensor scale within 2 float32 ulps: numerically nil, reported only
        else:
            untrained_changed.append(k)
    if untrained_changed:
        issues.append({"untrained_tensors_changed": untrained_changed[:20], "n": len(untrained_changed)})

    def qcfg(d):
        c = json.loads((d / "hf_quant_config.json").read_text())
        c.pop("producer", None)
        return c

    if qcfg(new.d) != qcfg(ref.d):
        issues.append({"hf_quant_config_differs": True})
    return {"tensors": len(nk), "untrained_identical": n_identical, "trained_tensors_changed": changed_trained,
            "scalar_ulp_drift": {"n": len(ulp_drift), "examples": ulp_drift[:5]}, "issues": issues}


# --------------------------------------------------------------------------------------------------------- train
def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("model", "init_export", "recipe", "train", "val", "head", "out"):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--tokens-per-step", type=int, default=16384)
    ap.add_argument("--micro-tokens", type=int, default=8192)
    ap.add_argument("--val-rows", type=int, default=2000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dry-run", type=int, default=0)
    ap.add_argument("--objective", choices=["readout", "distill"], default="readout",
                    help="readout: KL(corpus target || System 1 readout) (run 1). distill: full-vocab KL(BF16 teacher || "
                         "student) at every prompt position (NVIDIA QAD logits loss; run 2).")
    ap.add_argument("--amax-dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--max-grad-norm", type=float, default=0.0, help="0 = no clipping")
    ap.add_argument("--mem-fraction", type=float, default=0.0,
                    help="cap PyTorch's share of GPU memory so an overflow raises OOM (handled by halving the "
                         "micro-batch) instead of the Windows driver spilling into system RAM; 0 = off")
    ap.add_argument("--distill-val-rows", type=int, default=400)
    ap.add_argument("--movement-every", type=int, default=10,
                    help="log scale-movement stats every N steps (and at steps 1-3)")
    ap.add_argument("--quick-check", type=int, default=0,
                    help="N steps on a fixed 16-row batch, report movement + loss on that batch before/after, exit")
    a = ap.parse_args()
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    a.out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    if a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction, 0)
    t0 = time.time()
    from transformers import AutoTokenizer

    log(f"torch {torch.__version__} | GPU {torch.cuda.get_device_name(0)} | args {vars(a)}")
    tok = AutoTokenizer.from_pretrained(str(a.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    readout = Readout(a.head, tok)

    # Data first (CPU only): a data problem fails before the GPU is touched.
    n_steps = a.dry_run or a.max_steps
    train_rows = stratified(load_rows(a.train), n_steps * max(1, a.tokens_per_step // 100), rng)
    steps = make_steps(encode(train_rows, tok), a.tokens_per_step, random.Random(SEED + 2))
    val_enc = encode(stratified(load_rows(a.val), a.val_rows, random.Random(SEED + 1)), tok)
    log(f"train: {len(train_rows)} rows -> {len(steps)} steps of ~{a.tokens_per_step} tokens | validation: {len(val_enc)} rows")
    assert len(steps) >= n_steps, "not enough training rows for the step budget"

    model = load_student(a.model)
    log(f"model loaded in {time.time()-t0:.0f}s | GPU {torch.cuda.memory_allocated()/2**30:.1f} GiB")
    quantize_structure(model, a.recipe)
    ref = ExportReader(a.init_export)
    n = load_scales_from_export(model, ref)
    names = [nm for nm, _ in nvfp4_linears(model)]
    assert n == N_NVFP4_LINEARS == len(names), f"expected {N_NVFP4_LINEARS} NVFP4 linears, got {n}"
    sample = sample_linears(names)
    st = compare_to_export(fake_quant_weights(model, sample), ref)
    log(f"info: ModelOpt static fake-quant kernel vs B's export ({len(sample)} linears): {st}")
    params = enable_lsq(model, torch.bfloat16 if a.amax_dtype == "bf16" else torch.float32)
    chk = compare_to_export(fake_quant_weights(model, sample), ref)
    log(f"start point: LSQ step 0 (the training forward) vs B's export: {chk}")
    assert export_matches(chk), "LSQ step 0 does not reproduce candidate B"
    qs = enabled_quantizers(model)
    log(f"LSQ on {len(params)} quantizers, {sum(p.numel() for p in params)/1e6:.0f}M learnable {a.amax_dtype} amax; "
        f"step 0 == B | objective {a.objective} | {len(qs)} enabled quantizers | GPU "
        f"{torch.cuda.memory_allocated()/2**30:.1f} GiB")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0, fused=True)
    total = a.max_steps

    def lr_at(s: int) -> float:
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, (s - a.warmup) / max(1, total - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    micro = a.micro_tokens

    def mb_loss(mb):
        if a.objective == "distill":
            return distill_kl(model, mb, pad_id, dev, qs)
        return batch_kl(model, readout, mb, pad_id, dev)

    def run_step(step_rows) -> float:
        nonlocal micro
        denom = sum(len(i) for i, _ in step_rows) if a.objective == "distill" else len(step_rows)
        while True:
            try:
                opt.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for mb in micro_batches(step_rows, micro):
                    k, _ = mb_loss(mb)
                    (k / denom).backward()
                    loss_sum += float(k.detach())
                return loss_sum / denom
            except torch.cuda.OutOfMemoryError:
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if micro <= 1024:
                    raise
                micro //= 2
                log(f"OOM: micro-batch budget lowered to {micro} tokens (same step, same rows)")

    def evaluate(enc) -> float:
        model.eval()
        tot_kl, n_rows = 0.0, 0
        with torch.no_grad():
            for mb in micro_batches(enc, 2 * micro):
                k, m = batch_kl(model, readout, mb, pad_id, dev)
                tot_kl += float(k)
                n_rows += m
        model.train()
        return tot_kl / n_rows

    def evaluate_distill(enc) -> float:
        model.eval()
        tot_kl, n_tok = 0.0, 0
        with torch.no_grad():
            for mb in micro_batches(enc, micro):
                k, n = distill_kl(model, mb, pad_id, dev, qs)
                tot_kl += float(k)
                n_tok += n
        model.train()
        return tot_kl / max(1, n_tok)

    def save_best(step: int) -> None:
        torch.save({nm: p.detach().cpu() for nm, p in zip(names, params)}, a.out / "best-amax.pt")
        (a.out / "best-step.txt").write_text(str(step))

    qmods = [model.get_submodule(nm).weight_quantizer for nm in names]
    mv = Movement(qmods, params)
    if a.quick_check:
        # Fixed 16-row batch: train N steps on it and measure (a) movement and (b) the loss on that same batch before
        # and after. If the scales cannot move or the loss cannot fall on 16 rows, the full run cannot work either.
        fixed = steps[0][:16]
        model.eval()
        with torch.no_grad():
            k0, n0 = mb_loss(fixed)
        l0 = float(k0) / (n0 if a.objective == "distill" else len(fixed))
        model.train()
        log(f"QUICK-CHECK start: {a.objective}-KL on the fixed batch {l0:.6f}")
        for s in range(a.quick_check):
            for g in opt.param_groups:
                g["lr"] = a.lr
            loss = run_step(fixed)
            gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)) if a.max_grad_norm else float("nan")
            opt.step()
            with torch.no_grad():
                torch._foreach_clamp_min_(params, 1e-8)
            log(f"QUICK-CHECK step {s+1}: train {loss:.6f} grad-norm {gn:.3g} | movement {json.dumps(mv.report())}")
        model.eval()
        with torch.no_grad():
            k1, n1 = mb_loss(fixed)
        l1 = float(k1) / (n1 if a.objective == "distill" else len(fixed))
        r = mv.report(grads=False)
        ok = r["moved_scale"] > 0.01 and l1 < l0
        # bf16 contrast (same batch, same steps): would bf16 scales have moved? Shows whether the run-2a concern was real.
        log(f"QUICK-CHECK end: fixed-batch {a.objective}-KL {l0:.6f} -> {l1:.6f} ({100*(l1-l0)/l0:+.1f}%) | "
            f"FP8 block scales changed: {100*r['moved_scale']:.2f}% | {'QUICK-CHECK-OK' if ok else 'QUICK-CHECK-FAIL'}")
        with torch.no_grad():
            frozen_bf16 = moved_bf16 = total_n = 0
            for p, a0 in zip(params, mv.a0):
                d = (p.detach().float().cpu() - a0)
                a0b = a0.to(torch.bfloat16)
                moved = (a0 + d).to(torch.bfloat16) != a0b
                moved_bf16 += int(moved.sum())
                frozen_bf16 += int(((d != 0) & ~moved).sum())
                total_n += a0.numel()
        log(f"QUICK-CHECK bf16 contrast: of the fp32 updates applied, {100*frozen_bf16/max(1,total_n):.1f}% of blocks "
            f"would have been rounded away in bf16; {100*moved_bf16/max(1,total_n):.1f}% would have moved")
        return 0 if ok else 4

    logf = (a.out / "train-log.jsonl").open("a")
    model.train()
    te = time.time()
    if a.dry_run:
        v0 = evaluate(val_enc[:256])
        log(f"validation teacher-KL on 256 rows at step 0 (= B): {v0:.6f} in {time.time()-te:.0f}s "
            f"(2,000-row eval ~{(time.time()-te)*len(val_enc)/256/60:.1f} min)")
        best, best_step = v0, 0
    else:
        best, best_step = evaluate(val_enc), 0
        dv0 = evaluate_distill(val_enc[: a.distill_val_rows]) if a.objective == "distill" else float("nan")
        log(f"step 0 validation teacher-KL {best:.6f} | distill-KL vs BF16 {dv0:.6f} (start point = candidate B) "
            f"in {time.time()-te:.0f}s")
        logf.write(json.dumps({"step": 0, "val_kl": best, "val_distill_kl": dv0, "t": time.time() - t0}) + "\n")
        logf.flush()
        save_best(0)
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    deadline = t_train + a.max_hours * 3600
    eval_s = 0.0 if a.dry_run else (t_train - te)
    step, bad_evals, tokens_seen = 0, 0, 0
    for step_rows in steps:
        if step >= min(n_steps, total) or time.time() > deadline:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ts = time.time()
        loss = run_step(step_rows)
        gn = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)) if a.max_grad_norm else float("nan")
        opt.step()
        with torch.no_grad():
            torch._foreach_clamp_min_(params, 1e-8)
        step += 1
        tokens_seen += sum(len(i) for i, _ in step_rows)
        if not a.dry_run and step == 20:
            # Amendment A5: fit the cosine to the wall-clock budget, so the schedule anneals inside the 6 h cap.
            per_step = (time.time() - t_train) / step
            n_more = int((deadline - time.time()) / (per_step + eval_s / a.eval_every))
            fitted = min(a.max_steps, step + max(0, n_more))
            log(f"schedule fit (A5): {per_step:.1f} s/step, validation {eval_s:.0f} s -> cosine over {fitted} steps "
                f"(budget {a.max_steps})")
            total = fitted
        if step % 10 == 0 or a.dry_run or step <= 3:
            el = time.time() - t_train
            log(f"step {step}/{total} train {a.objective}-KL {loss:.6f} grad-norm {gn:.3g} lr {lr_at(step-1):.2e} | "
                f"{time.time()-ts:.1f}s/step {tokens_seen/el:.0f} tok/s | micro {micro} | peak GPU "
                f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        if step <= 3 or step % a.movement_every == 0:
            log(f"step {step} movement vs step 0: {json.dumps(mv.report())}")
        logf.write(json.dumps({"step": step, "train_kl": loss, "grad_norm": gn, "lr": lr_at(step - 1),
                               "t": time.time() - t0}) + "\n")
        if not a.dry_run and (step % a.eval_every == 0 or step >= total):
            v = evaluate(val_enc)
            dv = evaluate_distill(val_enc[: a.distill_val_rows]) if a.objective == "distill" else float("nan")
            logf.write(json.dumps({"step": step, "val_kl": v, "val_distill_kl": dv, "t": time.time() - t0}) + "\n")
            logf.flush()
            log(f"step {step} validation teacher-KL {v:.6f} | distill-KL vs BF16 {dv:.6f} (best {best:.6f} @ {best_step})")
            if v < best:
                best, best_step, bad_evals = v, step, 0
                save_best(step)
            else:
                bad_evals += 1
                if bad_evals >= a.patience:
                    log(f"early stop at step {step}: {a.patience} evaluations without improvement")
                    break
    el = time.time() - t_train
    log(f"training stopped after {step} steps, {el/3600:.2f} h, {tokens_seen/max(el, 1e-9):.0f} tok/s, "
        f"peak GPU {torch.cuda.max_memory_allocated()/2**30:.1f} GiB, final micro {micro}")
    if a.dry_run:
        est_h = a.max_steps * a.tokens_per_step / max(1.0, tokens_seen / el) / 3600
        log(f"estimate: {a.max_steps} steps would take {est_h:.1f} h (cap {a.max_hours} h)")
    else:
        if not math.isfinite(best):
            raise SystemExit("validation KL is not finite")
        state = torch.load(a.out / "best-amax.pt", weights_only=True)  # tensors only, written by this script
        with torch.no_grad():
            for nm, p in zip(names, params):
                p.copy_(state[nm].to(p.device))
        log(f"restored best amax from step {best_step} (validation teacher-KL {best:.6f})")
    del opt
    torch.cuda.empty_cache()
    model.gradient_checkpointing_disable()
    model.eval()
    fq_trained = fake_quant_weights(model, sample)  # the trained LSQ forward, before export packs the weights
    n_fold = fold_lsq(model)
    log(f"folded LSQ amax into {n_fold} static quantizers")
    exp_dir = a.out / ("dry-export" if a.dry_run else "export")
    te = time.time()
    export(model, a.model, exp_dir)
    log(f"exported to {exp_dir} in {time.time()-te:.0f}s")
    new = ExportReader(exp_dir)
    chk = compare_to_export(fq_trained, new)
    log(f"export vs trained LSQ fake-quant ({len(sample)} linears): {chk}")
    cmp = compare_exports(new, ref, names)
    log(f"export vs B's layout: {cmp}")
    blocking = [i for i in cmp["issues"] if "untrained_tensors_changed" in i or "dtype_or_shape" in i]
    ok = export_matches(chk) and not blocking
    summary = {"best_step": best_step, "best_val_kl": best, "steps_run": step, "train_hours": el / 3600,
               "tokens": tokens_seen, "micro_tokens": micro, "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
               "export": str(exp_dir), "export_vs_trained": chk, "export_vs_b": cmp, "export_ok": ok,
               "dry_run": bool(a.dry_run)}
    (a.out / "train-summary.json").write_text(json.dumps(summary, indent=1))
    if not ok:
        log("EXPORT-CHECK-FAILED " + json.dumps(summary))
        return 3
    log(("DRY-RUN-OK " if a.dry_run else "QAD-DONE ") + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
