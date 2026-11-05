#!/usr/bin/env python3
"""Quantize the Nemotron-3.5-Rerank-8B text tower with ModelOpt (issue #32) and export a vLLM-loadable checkpoint.

The checkpoint is Mistral3ForSequenceClassification (remote code, model.py): a Pixtral vision tower, a projector, a
bidirectional Ministral3 text tower and a mean-pooled score head. Only text-tower linears are quantized. The recipe's
default_disabled_quantizers already keeps the vision tower and projector BF16; the score head lives outside the text
tower and is never wrapped.

Calibration runs the WHOLE classifier on text-only inputs (the same forward the server runs: bidirectional attention,
mean pooling), so every text-tower quantizer sees serving-shaped activations. Export uses ModelOpt's unified HF exporter
(export_hf_checkpoint), which writes hf_quant_config.json and the quantized safetensors and copies the source's
non-model files. A post-step checks every tensor name and shape against the source, and that non-quantized tensors are
byte-identical to it.

Usage (inside the PTQ image, GPU): quantize_nr8.py RECIPE SOURCE_DIR CALIB_JSONL N_SAMPLES MAX_LEN OUT_DIR
Prints QUANT-OK <out> on success; exits non-zero otherwise."""
import importlib.util
import json
import sys
import time
from pathlib import Path, PurePosixPath

import modelopt.torch.quantization as mtq
import torch
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoConfig, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from model_forge.pipeline import CheckpointTree, OutputDir  # noqa: E402

recipe_path, src, calib_path, n_samples, max_len, out = sys.argv[1:7]
src, out, n_samples, max_len = Path(src), Path(out), int(n_samples), int(max_len)
t0 = time.time()

# The source files this script republishes (config.json, files the exporter skipped) are opened now, before anything is
# loaded or written: no symlinks except Hugging Face cache pointers, regular files only. They are copied from these
# descriptors below.
src_tree = CheckpointTree(src)
src_index = json.loads(src_tree.open("model.safetensors.index.json").read_bytes())
src_config = json.loads(src_tree.open("config.json").read_bytes())
shards = set(src_index["weight_map"].values())
src_top_files, src_top_dirs = src_tree.listdir()
src_rels = [name for name in src_top_files if not name.startswith(".")]
for name in src_top_dirs:
    if not name.startswith("."):
        src_rels += src_tree.walk(name)[0]
src_files = {rel: src_tree.open(rel) for rel in sorted(src_rels)
             if PurePosixPath(rel).name not in shards and PurePosixPath(rel).name != "model.safetensors.index.json"}

# The checkpoint's own model class (remote code), loaded by file so the exact pinned code runs.
spec = importlib.util.spec_from_file_location("nemo_rerank_model", src / "model.py")
assert spec is not None and spec.loader is not None, "cannot load the checkpoint's model.py"
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
cfg = AutoConfig.from_pretrained(src)
assert cfg.is_causal is False and cfg.text_config.is_causal is False, "expected a bidirectional text tower"
model = mod.Mistral3ForSequenceClassification.from_pretrained(src, config=cfg, dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained(src)
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s; params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
      flush=True)

# Calibration rows are pre-rendered with the checkpoint's chat template (query: ... passage: ...), no BOS/EOS.
rows = [json.loads(line) for line in Path(calib_path).read_bytes().split(b"\n") if line.strip()][:n_samples]
assert len(rows) == n_samples, f"calibration file has {len(rows)} rows, wanted {n_samples}"
encoded = [tok(r["text"], add_special_tokens=False, truncation=True, max_length=max_len, return_tensors="pt") for r in rows]
lens = sorted(int(e["input_ids"].shape[1]) for e in encoded)
print(f"calibration: {len(encoded)} rows, tokens p50 {lens[len(lens) // 2]} max {lens[-1]}", flush=True)


def forward_loop(m):
    with torch.no_grad():
        for i, e in enumerate(encoded):
            m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda())
            if (i + 1) % 64 == 0:
                print(f"  calib {i + 1}/{len(encoded)} ({time.time() - t0:.0f}s)", flush=True)


qcfg = load_recipe(recipe_path).quantize.model_dump()
print("recipe algorithm:", json.dumps(qcfg["algorithm"])[:240], flush=True)
model = mtq.quantize(model, qcfg, forward_loop=forward_loop)

# Coverage: count enabled quantizers by where they sit. Anything outside the text tower is a policy failure.
counts, outside = {}, []
for name, q in model.named_modules():
    if not isinstance(q, TensorQuantizer) or not q.is_enabled:
        continue
    where = "text" if ".language_model." in f".{name}" else "other"
    kind = next((k for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj") if k in name), "other")
    role = "weight" if name.endswith("weight_quantizer") else "input" if name.endswith("input_quantizer") else "other"
    fmt = "nvfp4" if getattr(q, "num_bits", None) == (2, 1) else str(getattr(q, "num_bits", "?"))
    counts[(where, kind, role, fmt)] = counts.get((where, kind, role, fmt), 0) + 1
    if where != "text":
        outside.append(name)
for k, v in sorted(counts.items()):
    print(f"  quantizers {k}: {v}", flush=True)
if outside:
    raise SystemExit(f"QUANT-FAILED: {len(outside)} quantizers outside the text tower, e.g. {outside[:3]}")

out.parent.mkdir(parents=True, exist_ok=True)
partial = out.with_name(out.name + ".partial")
# ModelOpt's exporter probes the model with a dummy forward, model(fake_input), to find linears that share an input
# (fused q/k/v, gate/up) so they get one input scale. NVIDIA's model.py requires attention_mask (mean pooling divides by
# its sum), so give the probe an all-ones mask by default. Only the probe calls without a mask; calibration and serving
# always pass one.
_forward = model.forward


def _forward_with_default_mask(input_ids=None, attention_mask=None, **kwargs):
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    return _forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


model.forward = _forward_with_default_mask
export_hf_checkpoint(model, export_dir=str(partial))
model.forward = _forward
print(f"exported to {partial} ({time.time() - t0:.0f}s)", flush=True)

# The exporter copies the source's non-model files, but not every one: on this checkpoint it skipped the
# sentence-transformers score-head file 2_Dense/model.safetensors. Copy every source file that is not a model shard and
# is missing from the export, byte for byte.
export_dir = OutputDir.create(partial)
for rel, member in src_files.items():
    if not (partial / rel).exists():
        export_dir.copy(rel, member)
        print(f"  copied missing source file {rel}", flush=True)

# transformers 5.18 re-serializes config.json on export: it drops default-valued keys and renames
# vision_config.rope_parameters.rope_type from "default" to "axial". vLLM and sentence-transformers read the upstream
# config, so put the source config.json back and keep only the export's quantization_config. The transformers copy is
# saved as config.transformers5.json and removed before release.
cfg_src = src_config
cfg_exp = json.loads((partial / "config.json").read_text())
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("QUANT-FAILED: export config.json has no quantization_config")
# vLLM 0.32 serves this checkpoint through its pooling adapter. The adapter keeps the text tower's language-model head as
# a placeholder module (`language_model.lm_head`, never loaded: a reranker has no LM head in its checkpoint) and the
# sequence-classification score head as `score` / `language_model.score`. Any of those not on the exclusion list gets an
# NVFP4 quant method; the lm_head placeholder then has no weights and vLLM fails with "Tensor.item() cannot be called
# on meta tensors" in process_weights_after_loading. Exclude all of them: none is quantized.
hq_path = partial / "hf_quant_config.json"
hq = json.loads(hq_path.read_text())
VLLM_HEADS = ("score", "language_model.score", "lm_head", "language_model.lm_head")
for container, key in ((hq["quantization"], "exclude_modules"), (qc, "ignore")):
    names = container.setdefault(key, [])
    for head in VLLM_HEADS:
        if head not in names:
            names.append(head)
hq_path.write_text(json.dumps(hq, indent=4) + "\n")
export_dir.copy("config.transformers5.json", CheckpointTree(partial).open("config.json"))
(partial / "config.json").write_text(json.dumps({**cfg_src, "quantization_config": qc}, indent=2) + "\n")
print(f"  restored source config.json (+ quantization_config, quant_algo {qc.get('quant_algo')}); "
      f"exclusions +{list(VLLM_HEADS)} for vLLM's pooling adapter", flush=True)
partial.rename(out)
print(f"QUANT-OK {out} ({time.time() - t0:.0f}s)", flush=True)
