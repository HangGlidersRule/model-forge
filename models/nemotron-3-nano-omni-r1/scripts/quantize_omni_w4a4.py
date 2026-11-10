#!/usr/bin/env python3
"""Issue #35: quantize Nemotron-3-Nano-Omni-30B-A3B-Reasoning (BF16, full VLM wrapper) with ModelOpt and export a
vLLM-loadable W4A4 checkpoint in NVIDIA's NVFP4 layout. Runs inside the omni PTQ image on the GPU host.

  1. Load the wrapper (trust_remote_code) in BF16 on the GPU; the vision tower, audio encoder and projectors are part of
     the forward, so calibration sees language-model activations for text, image and audio tokens as served.
  2. Calibrate with the recipe on CALIB_JSONL (rows from build_omni_calib.py): each row is rendered with the processor's
     chat template (thinking off) and fed through the full wrapper; image and audio rows carry their media.
  3. Export with ModelOpt's unified HF exporter, then restore the upstream wrapper config.json (keeping only the export's
     quantization_config) and copy every upstream non-weight file the export lacks, byte for byte.
  4. Prints QUANT-OK with the enabled quantizers by kind and format; fails closed on any quantizer outside the language
     model.
Usage: quantize_omni_w4a4.py RECIPE SOURCE CALIB_DIR N_ROWS MAX_TOKENS OUT_DIR"""
import collections
import json
import shutil
import sys
import time
from pathlib import Path

import modelopt.torch.quantization as mtq
import torch
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

recipe_path = sys.argv[1]
src, calib_dir, out = Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[6])
n_rows, max_tokens = int(sys.argv[4]), int(sys.argv[5])
t0 = time.time()
# Compatibility shim, no math change: the checkpoint's remote code (modeling_nemotron_h.py) calls
# create_causal_mask(input_embeds=..., cache_position=...), the signature before transformers renamed the argument to
# inputs_embeds and dropped cache_position (the mask is derived from position_ids / past_key_values instead). Translate
# the call; the module binds this wrapper when the remote code is imported below.
import inspect  # noqa: E402

import transformers.masking_utils as _mu  # noqa: E402

_ccm, _ccm_params = _mu.create_causal_mask, inspect.signature(_mu.create_causal_mask).parameters
if "input_embeds" not in _ccm_params:
    def _create_causal_mask_compat(*args, **kw):
        if "input_embeds" in kw:
            kw["inputs_embeds"] = kw.pop("input_embeds")
        if "cache_position" not in _ccm_params:
            kw.pop("cache_position", None)
        return _ccm(*args, **kw)
    _mu.create_causal_mask = _create_causal_mask_compat
    print("compat: create_causal_mask(input_embeds=, cache_position=) -> (inputs_embeds=)", flush=True)
# The wrapper's configuration.py defaults attn_implementation to flash_attention_2 (no flash-attn wheel exists for
# torch 2.15 / cu132), and the wrapper class does not declare sdpa support, so load it eager: exact attention, and only
# 6 of the 52 language-model layers are attention. The C-RADIO tower calls torch SDPA itself either way.
model = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="eager").eval()
proc = AutoProcessor.from_pretrained(src, trust_remote_code=True)
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s; params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
      flush=True)

# Memory. The image has no mamba_ssm kernels, so every Mamba layer runs the checkpoint's float32 torch_forward (chunked
# SSD scan). It keeps a (chunks x chunk x chunk x heads x state) float32 intermediate alive for the whole call: at
# chunk_size 128 that is ~14.5 GB for a 3,370-token image row, on top of 66 GB of weights, and the first full build
# died at row ~190 with the card at 95.6 of 97.9 GB ("CUDA driver error: device not ready"; under WSL the driver
# spills to system memory instead of raising OOM). The chunked scan is exact for any chunk size (only the summation
# order changes), and that intermediate scales linearly with it: 64 halves it, while the inter-chunk term (quadratic in
# the chunk count) stays smaller. Cap the allocator below the card so an oversized row raises a clean OOM (counted and
# skipped by forward_loop) instead of spilling.
_mixers = [m for m in model.modules() if type(m).__name__.endswith("Mamba2Mixer")]
for m in _mixers:
    m.chunk_size = 64
torch.cuda.set_per_process_memory_fraction(0.93)
print(f"memory: {len(_mixers)} Mamba mixers chunk_size 128 -> 64; allocator capped at 93% of "
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB; weights resident "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

# Float32 tensors. The checkpoint stores the routers' e_score_correction_bias (23 MoE layers) in float32, as NVIDIA's own
# FP8 and NVFP4 exports keep them; the router adds them to sigmoid scores before top-k. Loading with dtype=bfloat16
# rounds them (55.764542 -> 55.75: a 0.25 step at this magnitude), erasing the per-expert offsets that decide routing.
# Put every float32 language-model tensor back in float32 before calibration, so calibration routes tokens as served,
# and carry the exact upstream bytes into the export (restore_f32_in_export below).
import struct  # noqa: E402

from safetensors import safe_open  # noqa: E402


def _st_header(p: Path) -> tuple[dict, int]:
    with p.open("rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        h = json.loads(fh.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


_u_idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
F32: dict[str, torch.Tensor] = {}
for _shard in sorted(set(_u_idx.values())):
    _f32_names = [k for k, v in _st_header(src / _shard)[0].items() if v["dtype"] == "F32" and k.startswith("language_model.")]
    if _f32_names:
        with safe_open(str(src / _shard), framework="pt") as _fh:
            for k in _f32_names:
                F32[k] = _fh.get_tensor(k)
for name, t in F32.items():
    mod_name, attr = name.rsplit(".", 1)
    mod = model.get_submodule(mod_name)
    cur = getattr(mod, attr)
    if isinstance(cur, torch.nn.Parameter):
        setattr(mod, attr, torch.nn.Parameter(t.to(cur.device), requires_grad=False))
    else:
        mod._buffers[attr] = t.to(cur.device)
    if getattr(mod, attr).dtype != torch.float32:
        raise SystemExit(f"QUANT-FAILED: could not restore {name} to float32")
print(f"restored {len(F32)} float32 language-model tensors in memory: {sorted({n.rsplit('.', 1)[1] for n in F32})}", flush=True)


def restore_f32_in_export(root: Path) -> int:
    """Rewrite any export shard whose copy of a float32 tensor differs from the upstream bytes (dtype or value)."""
    idx_path = root / "model.safetensors.index.json"
    wm = json.loads(idx_path.read_text())["weight_map"] if idx_path.exists() else None
    shards = sorted(set(wm.values())) if wm else ["model.safetensors"]
    fixed = 0
    for shard in shards:
        p = root / shard
        h, off = _st_header(p)
        want = {k: F32[k].contiguous().numpy().tobytes() for k in h if k in F32}
        bad = []
        with p.open("rb") as fh:
            for k, b in want.items():
                s, e = h[k]["data_offsets"]
                fh.seek(off + s)
                if h[k]["dtype"] != "F32" or fh.read(e - s) != b:
                    bad.append(k)
        if not bad:
            continue
        new_h, chunks, pos = {}, [], 0
        for k, info in sorted(h.items(), key=lambda kv: kv[1]["data_offsets"][0]):
            if k in bad:
                n = len(want[k])
                new_h[k] = {"dtype": "F32", "shape": list(F32[k].shape), "data_offsets": [pos, pos + n]}
                chunks.append(("mem", want[k]))
            else:
                s, e = info["data_offsets"]
                n = e - s
                new_h[k] = {**info, "data_offsets": [pos, pos + n]}
                chunks.append(("file", (off + s, n)))
            pos += n
        blob = json.dumps(new_h, separators=(",", ":")).encode()
        blob += b" " * (-len(blob) % 8)
        tmp = p.with_name(p.name + ".f32fix")
        with p.open("rb") as fin, tmp.open("wb") as fout:
            fout.write(struct.pack("<Q", len(blob)))
            fout.write(blob)
            for kind, c in chunks:
                if kind == "mem":
                    fout.write(c)
                else:
                    fin.seek(c[0])
                    left = c[1]
                    while left:
                        buf = fin.read(min(left, 1 << 24))
                        fout.write(buf)
                        left -= len(buf)
        tmp.replace(p)
        fixed += len(bad)
    if fixed and idx_path.exists():
        idx = json.loads(idx_path.read_text())
        total = 0
        for shard in shards:
            total += sum(v["data_offsets"][1] - v["data_offsets"][0] for v in _st_header(root / shard)[0].values())
        idx.setdefault("metadata", {})["total_size"] = total
        idx_path.write_text(json.dumps(idx, indent=2) + "\n")
    return fixed

rows = [json.loads(line) for line in (calib_dir / "calib.jsonl").read_bytes().split(b"\n") if line.strip()]
# Interleave streams so a short smoke run (small N_ROWS) still covers text, image and audio.
by = collections.defaultdict(list)
for r in rows:
    by[r["stream"]].append(r)
order = []
while any(by.values()):
    for s in ("text", "image", "audio"):
        if by[s]:
            order.append(by[s].pop(0))
rows = order[:n_rows]


def encode(r):
    """One calibration row -> the inputs NVIDIA's own generate() takes (the wrapper's forward() handles images only;
    generate() assembles text, image and audio embeddings and runs the language model). Prefill only: max_new_tokens 1."""
    content, kwargs = [], {}
    if r["stream"] == "image":
        content.append({"type": "image"})
        kwargs["images"] = [Image.open(calib_dir / "media" / r["image"]).convert("RGB")]
    if r["stream"] == "audio":
        import soundfile as sf
        wav, sr = sf.read(calib_dir / "media" / r["audio"], dtype="float32")
        if sr != 16000:
            raise ValueError(f"sampling rate {sr}")
        content.append({"type": "audio"})
        kwargs["audio"] = [wav]
    content.append({"type": "text", "text": r["messages"][0]["content"]})
    msgs = [{"role": "user", "content": content}]
    if len(r["messages"]) > 1:
        msgs.append({"role": "assistant", "content": r["messages"][1]["content"]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    enc = proc(text=[text], return_tensors="pt", **kwargs)
    n = int(enc["input_ids"].shape[1])
    if n > max_tokens:
        if r["stream"] != "text":  # a media row's placeholder count must match its embeddings: skip, never truncate
            raise ValueError(f"{n} tokens > {max_tokens}")
        enc["input_ids"], enc["attention_mask"] = enc["input_ids"][:, :max_tokens], enc["attention_mask"][:, :max_tokens]
    out = {"input_ids": enc["input_ids"].to("cuda"), "attention_mask": enc["attention_mask"].to("cuda")}
    if "pixel_values" in enc:
        out["pixel_values"] = enc["pixel_values"].to("cuda", dtype=torch.bfloat16)
    if "sound_clips" in enc:
        out["sound_clips"] = enc["sound_clips"]
    return out


encoded, skipped = [], collections.Counter()
for r in rows:
    try:
        encoded.append((r["stream"], encode(r)))
    except Exception as e:  # noqa: BLE001
        skipped[f"{r['stream']}: {type(e).__name__}: {str(e)[:120]}"] += 1
streams = collections.Counter(s for s, _ in encoded)
lens = sorted(int(e["input_ids"].shape[1]) for _, e in encoded)
print(f"calibration: {len(encoded)} rows {dict(streams)}, tokens p50 {lens[len(lens) // 2]} max {lens[-1]}; skipped {dict(skipped)}",
      flush=True)
if not encoded or any(streams[s] == 0 for s in ("text", "image", "audio")):
    raise SystemExit(f"QUANT-FAILED: calibration must cover text, image and audio; got {dict(streams)}; skipped {dict(skipped)}")


def forward_loop(_quantized_root):
    """Calibration forward: always the full wrapper (vision tower, audio encoder, projectors and the language model), so
    the language model being calibrated sees text, image and audio tokens exactly as they arrive when served. A row that
    runs out of GPU memory is skipped and counted (max calibration keeps whatever the partial forward saw); more than 5%
    of rows skipped this way fails the build."""
    oom = collections.Counter()
    with torch.no_grad():
        for i, (stream, e) in enumerate(encoded):
            try:
                model.generate(**e, max_new_tokens=1, do_sample=False)
            except torch.cuda.OutOfMemoryError:
                oom[stream] += 1
                torch.cuda.empty_cache()
            if (i + 1) % 32 == 0:
                print(f"  calib {i + 1}/{len(encoded)} ({time.time() - t0:.0f}s) peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
    if oom:
        print(f"calibration OOM rows skipped: {dict(oom)}", flush=True)
    if sum(oom.values()) > 0.05 * len(encoded):
        raise SystemExit(f"QUANT-FAILED: {sum(oom.values())} of {len(encoded)} calibration rows ran out of memory")


# Calibration is prefill only. The wrapper's generate() assembles the text, image and audio embeddings exactly as NVIDIA
# serves them and then calls language_model.generate(..., use_cache=True); transformers 5.18's mask code expects cache
# methods the checkpoint's own NemotronHHybridDynamicCache lacks (get_query_offset). Replace that one call with a
# single cache-free forward of the language model on the assembled embeddings: the same prefill activations, no cache.
def _prefill_once(input_ids=None, inputs_embeds=None, attention_mask=None, **_):
    return model.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)


model.language_model.generate = _prefill_once


def _greedy(ids: torch.Tensor, steps: int) -> torch.Tensor:
    """Cache-free greedy decoding through the same forward path calibration uses."""
    eos = set(model.config.llm_config.eos_token_id if isinstance(model.config.llm_config.eos_token_id, list)
              else [model.config.llm_config.eos_token_id])
    for _ in range(steps):
        nxt = model.language_model(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt) in eos:
            break
    return ids


def sanity(tag: str) -> None:
    """Greedy answers through this exact transformers path (eager attention, the Mamba torch fallback, the mask shim)
    for two text prompts with known answers; fail closed if the BF16 model does not produce them, since a wrong forward
    would calibrate on activations the server never produces."""
    checks = (("What is the capital of Australia? Answer in one sentence.", "canberra"),
              ("What is 17 * 23? Answer with just the number.", "391"))
    got = []
    with torch.no_grad():
        for q, want in checks:
            text = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": q}]}], tokenize=False,
                                            add_generation_prompt=True, enable_thinking=False)
            enc = proc(text=[text], return_tensors="pt")
            n = enc["input_ids"].shape[1]
            ids = _greedy(enc["input_ids"].to("cuda"), 24)
            ans = proc.tokenizer.decode(ids[0][n:], skip_special_tokens=True)
            got.append((want, ans.strip()[:80]))
    print(f"sanity {tag}: {got}", flush=True)
    if tag == "bf16" and not all(w in a.lower() for w, a in got):
        raise SystemExit(f"QUANT-FAILED: the BF16 forward through this path does not answer correctly: {got}")


sanity("bf16")


qcfg = load_recipe(recipe_path).quantize.model_dump()
print("recipe algorithm:", json.dumps(qcfg["algorithm"])[:200], flush=True)
# Quantize the language model only, as ModelOpt's hf_ptq does for Nemotron VL models; calibration still drives the whole
# wrapper (forward_loop). The vision tower and audio encoder are then never converted to quant modules at all. Converting
# them is not harmless even with every quantizer disabled: ModelOpt 0.48.0 wraps transformers' Parakeet audio attention
# for KV-cache quantization, and that wrapper rejects Parakeet's keyword-argument attention call
# (TypeError: _QuantAttention._quantized_attention() missing 'query_states', 'key_states', 'value_states').
quantized_lm = mtq.quantize(model.language_model, qcfg, forward_loop=forward_loop)
if quantized_lm is not model.language_model:
    model.language_model = quantized_lm
    model.language_model.generate = _prefill_once
# Same two prompts through the fake-quantized model (information only: the gate is the served export). ModelOpt 0.48.0's
# Triton dynamic-FP4 kernel uses tl.make_block_ptr, which Triton 3.9 removed; route this forward through the CUDA extension.
import modelopt.torch.quantization.tensor_quant as _tq  # noqa: E402

_tq.DISABLE_TRITON_KERNEL = True
sanity("fakequant")

counts, outside = collections.Counter(), []
for name, q in model.named_modules():
    if not isinstance(q, TensorQuantizer) or not q.is_enabled:
        continue
    kind = next((k for k in ("shared_experts", "experts", "in_proj", "out_proj", "o_proj", "k_bmm", "v_bmm") if k in name), "other")
    role = "weight" if "weight_quantizer" in name else "input" if "input_quantizer" in name else "kv"
    fmt = "nvfp4" if getattr(q, "num_bits", None) == (2, 1) else "fp8" if getattr(q, "num_bits", None) == (4, 3) else str(getattr(q, "num_bits", "?"))
    counts[(kind, role, fmt)] += 1
    if not name.startswith("language_model."):
        outside.append(name)
for k, v in sorted(counts.items()):
    print(f"  quantizers {k}: {v}", flush=True)
if outside:
    raise SystemExit(f"QUANT-FAILED: {len(outside)} quantizers outside the language model, e.g. {outside[:3]}")

partial = out.with_name(out.name + ".partial")
shutil.rmtree(partial, ignore_errors=True)
export_hf_checkpoint(model, export_dir=str(partial))
print(f"exported ({time.time() - t0:.0f}s)", flush=True)
print(f"float32 tensors rewritten to the upstream bytes after export: {restore_f32_in_export(partial)} of {len(F32)}", flush=True)
shards = set(json.loads((src / "model.safetensors.index.json").read_text())["weight_map"].values())
for p in sorted(src.rglob("*")):
    rel = p.relative_to(src)
    if p.is_dir() or rel.name in shards or rel.name == "model.safetensors.index.json" or rel.parts[0] in (".git", "media") \
            or rel.name in ("SHA256SUMS.upstream", "SOURCE_REVISION", "README.md", ".gitattributes"):
        continue
    if not (partial / rel).exists():
        (partial / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, partial / rel)
cfg_src = json.loads((src / "config.json").read_text())
cfg_exp = json.loads((partial / "config.json").read_text())
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("QUANT-FAILED: export config.json has no quantization_config")
shutil.copyfile(partial / "config.json", partial / "config.transformers.json")
(partial / "config.json").write_text(json.dumps({**cfg_src, "quantization_config": qc}, indent=2) + "\n")
if (src / "generation_config.json").exists():
    shutil.copyfile(src / "generation_config.json", partial / "generation_config.json")
partial.rename(out)
print(f"QUANT-OK {out} ({time.time() - t0:.0f}s)", flush=True)
