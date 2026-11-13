#!/usr/bin/env python3
"""Ripple-65B Preview (#39): build one candidate (A, B or C) with ModelOpt 0.48.0 and export it (GPU host, PTQ image).

Path (posted pre-registration):
  load      the BF16 checkpoint with its own remote code under transformers 5.18, BF16, accelerate disk offload with
            an explicit device map: embeddings, final norm and lm_head on the GPU, the first (--gpu-gib - 2) / 1.51
            layers on the GPU, the next --cpu-gib / 1.51 on CPU, every other layer on disk (accelerate reads offloaded
            layers from the source safetensors). GPU at 512 rows (927,103 tokens): ~6 GiB resident weights + the
            calibrating layer 1.5 + cached inputs of two layers (hidden state and value residual per row, 2 x 9.5 GB
            each) ~38 + workspace: ~50-60 GB under a 0.94 memory-fraction cap. Host RAM: each committed checkpoint
            boundary moves one layer's cached inputs (~19 GB) to CPU before writing it, under the 40 GB cap.
  gate      known answers in this exact load path, one forward of a right-padded batch (no mask is needed: every real
            token precedes the padding): the top next token after "The capital of France is" must be " Paris" and
            after "The chemical symbol for gold is" " Au". Anything else stops the build (QUANT-FAILED) before
            calibration.
  stage 1   the candidate's decoder recipe (sha256 pinned), layerwise Local-Hessian calibration on the pinned
            calibration file (sha256 pinned): all 512 rows, or --rows N for a smoke (N/4 per source, file order).
            Resume: the recipe's layerwise block gets checkpoint_dir (and save_every) at run time, so an interrupted
            build restarts from the last committed layer; stale cached activations are pruned to the newest boundary.
            A crash after stage 1 completed (stage 2 or the export) rewinds the complete checkpoint to its newest saved
            boundary, so only the last few layers are recalibrated.
  stage 2   lm_head_w4a16_nvfp4_mse.yaml, data-free: set_quantizer_by_cfg + mtq.calibrate(lm_head, mse, no loop).
  census    enabled quantizers by kind/role/format must equal the candidate's plan (A 1,281, B 1,041, C 481).
  export    export_hf_checkpoint (ModelOpt's streaming writer for an offloaded model) into OUT.partial; then the
            source config.json is restored with the export's quantization_config (MIXED_PRECISION, one entry per
            quantized linear) added, the remote code, template and tokenizer files are copied byte for byte, and
            OUT.partial is renamed to OUT. eval/validate_export.py then checks it (the window runs it).
Writes OUT.build.json (pins, rows, token stats, per-layer seconds, peak GPU memory, quantizer census, timings).
Usage (PTQ image, GPU): quantize_ripple.py LETTER SOURCE_DIR CALIB_JSONL OUT_DIR WORK_DIR [--rows N] [--gpu-gib G]
       [--cpu-gib C] [--save-every K]
CPU rehearsal (NAS, no GPU): --cpu-smoke builds a tiny 2-layer checkpoint from the real config (random weights) in
WORK_DIR, swaps every NVFP4 rule for FP8 (ModelOpt's NVFP4 fake quant is CUDA-only), offloads layer 1 to disk, and runs
the same stages; --stop-after-layer K exits after layer K's commit (to rehearse a resume), --stop-after-stage1 exits
right after stage 1 (to rehearse a crash before the export). Prints QUANT-OK <out> on success."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

FAMILY = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FAMILY / "eval"))
import ripple_layout as lay  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("letter")
ap.add_argument("source", type=Path)
ap.add_argument("calib", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("work", type=Path)
ap.add_argument("--rows", type=int, default=lay.CALIB_ROWS)
ap.add_argument("--gpu-gib", type=float, default=8.0)
ap.add_argument("--cpu-gib", type=float, default=4.0)
ap.add_argument("--save-every", type=int, default=4)
ap.add_argument("--recipe-dir", type=Path, default=FAMILY.parent.parent / lay.RECIPE_DIR)
ap.add_argument("--cpu-smoke", action="store_true")
ap.add_argument("--stop-after-layer", type=int, default=None)
ap.add_argument("--stop-after-stage1", action="store_true")
args = ap.parse_args()
letter = lay.check_letter(args.letter)
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)


def fail(msg: str) -> None:
    print(f"QUANT-FAILED: {msg}", flush=True)
    sys.exit(1)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


import modelopt.torch.quantization as mtq  # noqa: E402
import modelopt.torch.quantization.tensor_quant as tensor_quant  # noqa: E402
import torch  # noqa: E402
from modelopt.recipe import load_recipe  # noqa: E402
from modelopt.torch.export import export_hf_checkpoint  # noqa: E402
from modelopt.torch.quantization.conversion import set_quantizer_by_cfg  # noqa: E402
from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402
from modelopt.torch.quantization.utils import layerwise_calib as lwc  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# ModelOpt 0.48.0's Triton dynamic-FP4 fake-quant kernel does not compile under torch 2.15's Triton 3.9; the layerwise
# capture of each next layer's inputs runs the calibrated layer with fake quant, so route that op through ModelOpt's
# CUDA extension from the start (same op; the static weight fake quant keeps its own kernel).
tensor_quant.DISABLE_TRITON_KERNEL = True

# ---- inputs and pins --------------------------------------------------------------------------------------------------
recipe_name, recipe_pin = lay.RECIPES[letter]
recipe_path = args.recipe_dir / recipe_name
lm_path = args.recipe_dir / lay.LM_HEAD_RECIPE[0]
for p, want in ((recipe_path, recipe_pin), (lm_path, lay.LM_HEAD_RECIPE[1])):
    if sha(p) != want:
        fail(f"{p.name} sha256 {sha(p)} is not the pin {want}")
calib_bytes = args.calib.read_bytes()
calib_sha = hashlib.sha256(calib_bytes).hexdigest()
if not args.cpu_smoke and calib_sha != lay.CALIB_SHA256:
    fail(f"calibration file sha256 {calib_sha} is not the pin {lay.CALIB_SHA256}")
rows_all = [json.loads(line) for line in calib_bytes.split(b"\n") if line.strip()]
take = lay.calib_take(rows_all, args.rows)
if not args.cpu_smoke and args.rows >= lay.CALIB_ROWS and len(take) != lay.CALIB_ROWS:
    fail(f"{len(take)} calibration rows, expected {lay.CALIB_ROWS}")
args.work.mkdir(parents=True, exist_ok=True)
run_key = hashlib.sha256(json.dumps([letter, recipe_pin, calib_sha, args.rows, args.cpu_smoke]).encode()).hexdigest()
work = args.work / f"{letter}-{args.rows}-{run_key[:12]}"
work.mkdir(parents=True, exist_ok=True)
ckpt_dir = work / "layerwise-ckpt"

qcfg = load_recipe(recipe_path).quantize.model_dump()
lmq = load_recipe(lm_path).quantize.model_dump()
if args.cpu_smoke:
    # CPU: ModelOpt's NVFP4 fake quant is CUDA/Triton-only; keep the algorithm block, swap NVFP4 rules for FP8.
    fp8 = next(e["cfg"] for e in qcfg["quant_cfg"] if isinstance(e.get("cfg"), dict)
               and tuple(e["cfg"].get("num_bits") or ()) == (4, 3)) if letter != "C" else \
        {"num_bits": (4, 3), "axis": None}
    for cfg in (qcfg, lmq):
        for e in cfg["quant_cfg"]:
            if isinstance(e.get("cfg"), dict) and tuple(e["cfg"].get("num_bits") or ()) == (2, 1):
                e["cfg"] = dict(fp8)
    lmq["algorithm"] = {"method": "max"}
lw = qcfg["algorithm"].setdefault("layerwise", {})
lw["checkpoint_dir"] = str(ckpt_dir)
lw["save_every"] = args.save_every

# ---- source: the real checkpoint, or (CPU rehearsal) a tiny one built from the real config -------------------------
src = args.source
if args.cpu_smoke:
    tiny = work / "tiny-src"
    if not (tiny / "model.safetensors").exists() and not (tiny / "model.safetensors.index.json").exists():
        cfg_t = AutoConfig.from_pretrained(src, trust_remote_code=True, num_hidden_layers=2,
                                           layer_types=["sliding_attention", "full_attention"], hidden_size=256,
                                           intermediate_size=512, num_attention_heads=4, num_key_value_heads=2,
                                           head_dim=64, sliding_window=8, embed_scale=16.0)
        torch.manual_seed(1234)
        m0 = AutoModelForCausalLM.from_config(cfg_t, trust_remote_code=True, dtype=torch.float32)
        with torch.no_grad():
            for n, p in m0.named_parameters():
                if n.endswith("norm.weight"):
                    p.fill_(1.0)
                elif "value_res_lambda" in n:
                    p.fill_(0.3)
                else:
                    p.normal_(0.0, 0.02)
        m0.to(torch.bfloat16).save_pretrained(tiny, safe_serialization=True)
        for f in lay.COPIED_FILES:
            if (src / f).exists():
                shutil.copyfile(src / f, tiny / f)
        del m0
    src = tiny

cfg_src = json.loads((src / "config.json").read_text())
n_layers = int(cfg_src["num_hidden_layers"])
if not args.cpu_smoke:
    for f, want in lay.SOURCE_FILE_PINS.items():
        if sha(src / f) != want:
            fail(f"source {f} is not the pinned file")
tok = AutoTokenizer.from_pretrained(src)
DEV = "cpu" if args.cpu_smoke else "cuda"
# Explicit device map (deterministic, and the same shape the CPU rehearsal runs): embeddings, final norm and lm_head
# resident (stage 2 calibrates lm_head with its weight in place), the first layers on the GPU up to --gpu-gib, the next
# on CPU up to --cpu-gib, every other layer on disk (accelerate reads those from the source safetensors per use).
LAYER_GIB = 1.51  # one decoder layer in BF16 (0.80B linear parameters + norms), from the safetensors headers
if not args.cpu_smoke:
    if not torch.cuda.is_available():
        fail("no CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.94, 0)  # an overflow raises instead of paging into shared memory
    n_gpu = max(0, int((args.gpu_gib - 2.0) // LAYER_GIB))
    n_cpu = max(0, int(args.cpu_gib // LAYER_GIB))
    home: object = 0
else:
    n_gpu, n_cpu, home = 0, 1, "cpu"
device_map: dict[str, object] = {"model.embed_tokens": home, "model.norm": home, "lm_head": home}
for i in range(n_layers):
    device_map[f"model.layers.{i}"] = home if i < n_gpu else ("cpu" if i < n_gpu + n_cpu else "disk")
offload = work / "offload"
offload.mkdir(exist_ok=True)
model = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True, dtype=torch.bfloat16, device_map=device_map,
                                             offload_folder=str(offload)).eval()
placement: dict[str, int] = {}
for v in (getattr(model, "hf_device_map", None) or {}).values():
    placement[str(v)] = placement.get(str(v), 0) + 1
log(f"loaded {type(model).__name__} ({n_layers} layers) device map {placement}; "
    f"gpu {torch.cuda.memory_allocated() / 2**30 if DEV == 'cuda' else 0:.1f} GiB")


def first_device():
    return model.model.embed_tokens.weight.device if model.model.embed_tokens.weight.device.type != "meta" else DEV


# ---- known-answer gate --------------------------------------------------------------------------------------------------
enc = [tok(p, add_special_tokens=False)["input_ids"] for p, _ in lay.KNOWN_ANSWERS]
width = max(len(e) for e in enc)
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
batch = torch.tensor([e + [pad] * (width - len(e)) for e in enc])
with torch.no_grad():
    logits = model(input_ids=batch.to(first_device()), use_cache=False).logits.float()
known = {}
for (p, want), e, lg in zip(lay.KNOWN_ANSWERS, enc, logits):
    lp = torch.log_softmax(lg[len(e) - 1], -1)
    top = torch.topk(lp, 3)
    known[p] = [(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values, top.indices)]
    if known[p][0][0] != want:
        if args.cpu_smoke:
            log(f"known answer (tiny random model, not gated): {p!r} -> {known[p][0]}")
            continue
        fail(f"known answer for {p!r} is {known[p][0]}, not {want!r} (this load path computes something else)")
log(f"KNOWN-ANSWER-OK {json.dumps(known)}" if not args.cpu_smoke else "KNOWN-ANSWER path ran (CPU rehearsal)")

# ---- calibration rows -----------------------------------------------------------------------------------------------------
encoded = [torch.tensor(tok(r["text"], add_special_tokens=False)["input_ids"])[None] for r in take]
if args.cpu_smoke:
    encoded = [e[:, :256] for e in encoded]
lens = sorted(int(e.shape[1]) for e in encoded)
log(f"calibration: {len(encoded)} rows ({json.dumps({k: sum(1 for r in take if r['source'] == k) for k in lay.SOURCES_IN_FILE_ORDER})}), "
    f"tokens {sum(lens)} (p50 {lens[len(lens) // 2]} p90 {lens[int(len(lens) * 0.9)]} max {lens[-1]}), file {calib_sha[:16]}")
loop_calls = [0]


def forward_loop(m):
    loop_calls[0] += 1
    dev = first_device()
    with torch.no_grad():
        for e in encoded:
            m(input_ids=e.to(dev), use_cache=False)


# Per-layer progress lines and pruning of stale cached activations (save_layer_state keeps every boundary's
# next_inputs.pt otherwise: ~19 GB each at 512 rows). Only the newest committed boundary is kept; the final layer
# writes no next_inputs, so the last boundary before it stays for a rewind after a crash past stage 1.
layer_times: list[float] = []
_last = [time.time()]
_orig_save = lwc._CheckpointState.save


def _save(self, layer_idx, model_, layers, next_layer_inputs=None):
    _orig_save(self, layer_idx, model_, layers, next_layer_inputs)
    now = time.time()
    layer_times.append(now - _last[0])
    _last[0] = now
    if next_layer_inputs is not None and ((layer_idx + 1) % self.save_every == 0):
        self._prune_stale_next_inputs(keep=layer_idx)
    peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
    recent = layer_times[-8:]
    rate = sum(recent) / len(recent)
    print(f"LAYER-DONE {layer_idx + 1}/{len(layers)} {layer_times[-1]:.1f}s (recent {rate:.1f}s/layer, ETA "
          f"{rate * (len(layers) - layer_idx - 1) / 60:.1f} min) gpu peak {peak:.1f} GiB", flush=True)
    if args.stop_after_layer is not None and layer_idx >= args.stop_after_layer:
        print(f"STOPPED after layer {layer_idx} (rehearsal of an interrupted build)", flush=True)
        os._exit(75)


lwc._CheckpointState.save = _save

# ---- stage 1 -------------------------------------------------------------------------------------------------------------
# Resume: ModelOpt resumes from the newest committed boundary of an incomplete checkpoint. A COMPLETE manifest (stage 1
# finished, then the process died in stage 2 or the export) would restart at layer 0, so it is rewound to the newest
# boundary that still holds next_inputs.pt: only the layers after it are recalibrated; the rest restore from their files.
t_cal = time.time()
manifest = lwc._read_manifest(str(ckpt_dir))
if manifest and manifest.get("last_completed_layer") is not None and \
        manifest["last_completed_layer"] + 1 >= int(manifest.get("num_layers") or 0):
    kept = sorted(int(p.parent.name.split("_")[1]) for p in ckpt_dir.glob("layer_*/next_inputs.pt"))
    if kept:
        lwc._write_manifest(str(ckpt_dir), kept[-1], int(manifest["num_layers"]),
                            save_every=int(manifest.get("save_every") or args.save_every),
                            calib_mutates_weights=bool(manifest.get("calib_mutates_weights")),
                            save_layer_state=bool(manifest.get("save_layer_state", True)))
        log(f"stage 1 had completed; rewound its checkpoint to layer {kept[-1]} (the newest saved boundary)")
    else:
        shutil.rmtree(ckpt_dir)
        log("stage 1 had completed but no boundary inputs remain; recalibrating from layer 0")
info = lwc.detect_resume_point(str(ckpt_dir))
resumed_at = info[0] if info else 0
if info:
    log(f"resuming stage 1 at layer {info[0]}/{n_layers} from {ckpt_dir.name}")
log(f"stage 1: {recipe_name} ({json.dumps(qcfg['algorithm'])[:220]})")
_last[0] = time.time()  # per-layer times start here, not at the load
model = mtq.quantize(model, qcfg, forward_loop=forward_loop)
log(f"STAGE1-OK in {time.time() - t_cal:.0f}s ({loop_calls[0]} forward-loop passes, started at layer {resumed_at})")
if args.stop_after_stage1:
    print("STOPPED after stage 1 (rehearsal of a crash before the export)", flush=True)
    os._exit(75)

# ---- stage 2: lm_head W4A16 NVFP4, data-free ------------------------------------------------------------------------------
set_quantizer_by_cfg(model, lmq["quant_cfg"])
mtq.calibrate(model.lm_head, lmq["algorithm"], forward_loop=None)
log(f"STAGE2-OK lm_head ({json.dumps(lmq['algorithm'])})")

# ---- census ---------------------------------------------------------------------------------------------------------------
census: dict[str, int] = {}
no_amax = []
for name, q in model.named_modules():
    if not isinstance(q, TensorQuantizer) or not q.is_enabled:
        continue
    lin = name.rsplit(".", 1)[0]
    kind = "lm_head" if lin == "lm_head" else ("attn" if ".self_attn." in name else "mlp" if ".mlp." in name else "other")
    role = "weight" if name.endswith("weight_quantizer") else "input" if name.endswith("input_quantizer") else "other"
    nb = getattr(q, "num_bits", None)
    fmt = "nvfp4" if nb == (2, 1) else "fp8" if nb == (4, 3) else str(nb)
    census[f"{kind}|{role}|{fmt}"] = census.get(f"{kind}|{role}|{fmt}", 0) + 1
    if name.endswith("input_quantizer") and getattr(q, "_dynamic", False) is False:
        a = getattr(q, "amax", None)
        if a is None or not bool(torch.isfinite(a).all()) or not bool((a > 0).all()):
            no_amax.append(name)
total = sum(census.values())
want_total = lay.expected_quantizer_total(letter, n_layers)
log(f"quantizers: {total} enabled {json.dumps(census, sort_keys=True)} (plan {want_total})")
if total != want_total or any(k.startswith("other") for k in census):
    fail(f"enabled quantizers {total} differ from {letter}'s plan {want_total}: {census}")
if no_amax:
    fail(f"{len(no_amax)} static input quantizers without a finite positive amax: {no_amax[:3]}")

# ---- export ---------------------------------------------------------------------------------------------------------------
partial = args.out.with_name(args.out.name + ".partial")
shutil.rmtree(partial, ignore_errors=True)
t_ex = time.time()
# ModelOpt 0.48.0's streaming export (taken for an offloaded model) runs the per-module export handler, which packs a
# quantized linear into its export format, only for decoder layers and for non-decoder modules whose weights are still
# offloaded (requires_weight_materialization). A quantized lm_head resident on the GPU is in neither set, so the
# streaming path wrote it as the original BF16 tensor with no scales (the #39 smoke, 2026-11-12: lm_head.weight BF16
# [100352, 5120], no weight_scale / weight_scale_2), while hf_quant_config still listed it W4A16_NVFP4. Packing it before
# export_hf_checkpoint breaks the exporter's own dummy forward (requantize_resmooth_fused_llm_layers runs the whole model,
# and a packed lm_head no longer matmuls). So the pack runs inside the streaming export, right after it has computed the
# quant config (after the dummy forward) and before any tensor is streamed: get_quant_config, as bound in the streaming
# module, is wrapped. The streaming path then writes the packed tensors as resident parameters and buffers.
import modelopt.torch.export.unified_export_hf_streaming as _ehs  # noqa: E402
from modelopt.torch.export.registry import ExportContext  # noqa: E402
from modelopt.torch.export.unified_export_hf import _dispatch_export_handler, _resolve_export_dtype  # noqa: E402
from modelopt.torch.models import hf_model_type  # noqa: E402
from modelopt.torch.quantization.utils.core_utils import (  # noqa: E402
    module_name_maps,
    requires_weight_materialization,
)

pre_packed: list[str] = []
_orig_get_quant_config = _ehs.get_quant_config


def _get_quant_config_then_pack(m, *a, **kw):
    qc_ = _orig_get_quant_config(m, *a, **kw)
    ctx_ = ExportContext(model=m, dtype=_resolve_export_dtype(m, None), is_modelopt_qlora=False,
                         model_type=hf_model_type(m))
    names_ = module_name_maps(m)
    owned = {id(x) for layer in lwc.LayerActivationCollector.get_decoder_layers(m) for x in layer.modules()}
    for name, module in m.named_modules():
        if id(module) in owned or requires_weight_materialization(module, m, names_):
            continue
        wq = getattr(module, "weight_quantizer", None)
        w = getattr(module, "weight", None)
        if isinstance(wq, TensorQuantizer) and wq.is_enabled and w is not None and not w.is_meta:
            _dispatch_export_handler(name, module, ctx_)
            pre_packed.append(name)
    return qc_


_ehs.get_quant_config = _get_quant_config_then_pack
try:
    export_hf_checkpoint(model, export_dir=str(partial))
finally:
    _ehs.get_quant_config = _orig_get_quant_config
log(f"resident quantized linears outside the decoder layers packed inside the export: {pre_packed}")
if lay.PLANS[letter]["lm_head"] and "lm_head" not in pre_packed:
    fail(f"lm_head is quantized in the plan ({lay.PLANS[letter]['lm_head']}) but was not packed: {pre_packed}")
log(f"exported to {partial.name} in {time.time() - t_ex:.0f}s")
qc = json.loads((partial / "config.json").read_text()).get("quantization_config")
if not qc:
    fail("the export's config.json has no quantization_config")
shutil.copyfile(partial / "config.json", partial / "config.transformers5.json")
exp_cfg = json.loads((partial / "config.json").read_text())
new_cfg = {**cfg_src, "quantization_config": qc}
if "dtype" in exp_cfg and "dtype" not in new_cfg:
    new_cfg["dtype"] = exp_cfg["dtype"]
(partial / "config.json").write_text(json.dumps(new_cfg, indent=2) + "\n")
copied = []
for f in lay.COPIED_FILES:
    if (src / f).exists() and (not (partial / f).exists() or sha(partial / f) != sha(src / f)):
        shutil.copyfile(src / f, partial / f)
        copied.append(f)
# The NAS store's own metadata describes the BF16 source, not this export: drop it if the exporter copied it.
dropped = [f for f in ("SHA256SUMS.upstream", "SOURCE_REVISION", "SUBSET") if (partial / f).exists()]
for f in dropped:
    (partial / f).unlink()
hq = json.loads((partial / "hf_quant_config.json").read_text())
log(f"store metadata not carried: {dropped}")
log(f"config.json restored from the source + quantization_config ({qc.get('quant_algo')}, "
    f"{len(qc.get('quantized_layers') or {})} layers); hf_quant_config {hq['quantization'].get('quant_algo')}; "
    f"copied from source: {copied}")
shutil.rmtree(args.out, ignore_errors=True)
partial.rename(args.out)
shutil.rmtree(ckpt_dir, ignore_errors=True)  # the export is complete; the calibration checkpoint is no longer needed
shutil.rmtree(offload, ignore_errors=True)
peak = torch.cuda.max_memory_allocated() / 2**30 if DEV == "cuda" else 0.0
rec = {"candidate": letter, "product_id": lay.PRODUCT_IDS[letter], "recipe": recipe_name, "recipe_sha256": recipe_pin,
       "lm_head_recipe_sha256": lay.LM_HEAD_RECIPE[1], "calibration_sha256": calib_sha, "rows": len(encoded),
       "tokens": {"total": sum(lens), "p50": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)], "max": lens[-1]},
       "device_map_counts": placement, "known_answer": known, "quantizers": census, "quantizers_total": total,
       "stage1_resumed_at_layer": resumed_at, "layer_seconds": [round(x, 1) for x in layer_times],
       "seconds_per_layer_mean": round(sum(layer_times) / len(layer_times), 1) if layer_times else None,
       "seconds_per_row_per_layer": round(sum(layer_times) / len(layer_times) / len(encoded), 3) if layer_times else None,
       "stage1_seconds": round(time.time() - t_cal), "export_seconds": round(time.time() - t_ex),
       "gpu_peak_gib": round(peak, 1), "cpu_smoke": args.cpu_smoke, "seconds": round(time.time() - t0)}
args.out.with_name(args.out.name + ".build.json").write_text(json.dumps(rec, indent=1) + "\n")
print(f"QUANT-OK {args.out} | {letter} rows {len(encoded)} | {rec['seconds_per_layer_mean']} s/layer | "
      f"peak {rec['gpu_peak_gib']} GiB | {rec['seconds']} s", flush=True)
