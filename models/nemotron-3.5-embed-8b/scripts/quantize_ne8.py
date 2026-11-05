#!/usr/bin/env python3
"""Quantize the Nemotron-3.5-Embed-8B text tower with ModelOpt (issue #33) and export a vLLM-loadable checkpoint.

The checkpoint is a plain Mistral3Model (no remote code, no head): a Pixtral vision tower, a projector and a bidirectional
Ministral3 text tower; embeddings are the mean-pooled, L2-normalized last hidden state (sentence-transformers modules).
Only text-tower linears are quantized; the recipe's default_disabled_quantizers keeps the vision tower and projector BF16.

Before quantizing, the script proves the loaded text tower is bidirectional (changing the LAST token must change the
FIRST token's hidden state) and writes reference embeddings for 16 fixed calibration texts: BF16, and again after
calibration (fake-quantized), so the served export can be checked against both. Calibration runs the model on
chat-template-rendered text ("query: ..." / "passage: ...", no BOS/EOS), the same tokens the server sees.

Export uses ModelOpt's unified HF exporter, then repairs what the exporter changes outside the weights (the same repairs
as Nemotron-3.5-Rerank-8B, #32): source files it skips are copied byte for byte, the upstream config.json is restored
with only quantization_config added, and vLLM's pooling-adapter lm_head placeholder is excluded.

Usage (inside the PTQ image, GPU): quantize_ne8.py RECIPE SOURCE_DIR CALIB_JSONL N_SAMPLES MAX_LEN OUT_DIR
Prints QUANT-OK <out> on success; exits non-zero otherwise. Writes OUT_DIR.ref.json (reference embeddings)."""
import json
import sys
import time
from pathlib import Path, PurePosixPath

import modelopt.torch.quantization as mtq
import torch
import torch.nn.functional as F
from modelopt.recipe import load_recipe
from modelopt.torch.export import export_hf_checkpoint
from modelopt.torch.quantization.nn import TensorQuantizer
from transformers import AutoConfig, AutoModel, AutoTokenizer

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

cfg = AutoConfig.from_pretrained(src)
assert cfg.is_causal is False and cfg.text_config.is_causal is False, "expected a bidirectional text tower"
model = AutoModel.from_pretrained(src, config=cfg, dtype=torch.bfloat16).cuda().eval()
tok = AutoTokenizer.from_pretrained(src)
print(f"loaded {type(model).__name__} in {time.time() - t0:.0f}s; params {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B",
      flush=True)
assert type(model).__name__ == "Mistral3Model", type(model).__name__


def hidden(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state


# Bidirectionality: with a causal mask the first position cannot see the last token.
with torch.no_grad():
    e = tok("passage: the quick brown fox jumps over the lazy dog", add_special_tokens=False, return_tensors="pt")
    ids2 = e["input_ids"].clone()
    ids2[0, -1] = tok("cat", add_special_tokens=False)["input_ids"][0]
    d0 = (hidden(e["input_ids"], e["attention_mask"])[0, 0] - hidden(ids2, e["attention_mask"])[0, 0]).abs().max().item()
print(f"bidirectional check: first-token hidden state moved {d0:.4g} when the last token changed", flush=True)
assert d0 > 0, "QUANT-FAILED: the text tower is causal as loaded (first position ignores the last token)"

rows = [json.loads(line) for line in Path(calib_path).read_bytes().split(b"\n") if line.strip()][:n_samples]
assert len(rows) == n_samples, f"calibration file has {len(rows)} rows, wanted {n_samples}"
encoded = [tok(r["text"], add_special_tokens=False, truncation=True, max_length=max_len, return_tensors="pt") for r in rows]
lens = sorted(int(x["input_ids"].shape[1]) for x in encoded)
print(f"calibration: {len(encoded)} rows, tokens p50 {lens[len(lens) // 2]} max {lens[-1]}", flush=True)
all_rows = [json.loads(line) for line in Path(calib_path).read_bytes().split(b"\n") if line.strip()]
ref_texts = [r["text"] for r in all_rows if r["text"].startswith("query: ")][:8] + \
            [r["text"] for r in all_rows if r["text"].startswith("passage: ")][:8]


def embed_refs() -> list[list[float]]:
    out_v = []
    with torch.no_grad():
        for t in ref_texts:
            x = tok(t, add_special_tokens=False, return_tensors="pt")
            h = hidden(x["input_ids"], x["attention_mask"]).float()
            m = x["attention_mask"].cuda()[..., None].float()
            out_v.append(F.normalize((h * m).sum(1) / m.sum(1), dim=-1)[0].cpu().tolist())
    return out_v


ref = {"texts": ref_texts, "bf16": embed_refs()}


def forward_loop(m):
    with torch.no_grad():
        for i, x in enumerate(encoded):
            m(input_ids=x["input_ids"].cuda(), attention_mask=x["attention_mask"].cuda())
            if (i + 1) % 64 == 0:
                print(f"  calib {i + 1}/{len(encoded)} ({time.time() - t0:.0f}s)", flush=True)


qcfg = load_recipe(recipe_path).quantize.model_dump()
print("recipe algorithm:", json.dumps(qcfg["algorithm"])[:240], flush=True)
model = mtq.quantize(model, qcfg, forward_loop=forward_loop)
# ModelOpt 0.48.0's Triton dynamic FP4 fake-quant kernel (fp4_fake_quant_block, kernels/quantization/gemm/
# fp4_kernel_hopper.py) uses tl.make_block_ptr, which Triton 3.9 (shipped with torch 2.15) removed: "Block pointers have
# been removed in favor of the tensor descriptor API". Calibration never calls it (the reranker build, #32, ran no forward
# after quantize); this script's post-calibration fake-quant reference forward does. Route that through ModelOpt's CUDA
# extension instead (tensor_quant.DISABLE_TRITON_KERNEL is read at call time). Export is unaffected.
import modelopt.torch.quantization.tensor_quant as tensor_quant  # noqa: E402

tensor_quant.DISABLE_TRITON_KERNEL = True
ref["fakequant"] = embed_refs()
cos = [sum(a * b for a, b in zip(x, y)) for x, y in zip(ref["bf16"], ref["fakequant"])]
print(f"fake-quant vs BF16 reference cosine: min {min(cos):.4f} mean {sum(cos) / len(cos):.4f}", flush=True)

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
_forward = model.forward


def _forward_with_default_mask(input_ids=None, attention_mask=None, **kwargs):
    # ModelOpt's export probe calls the model without a mask; give it an all-ones one (calibration always passes one).
    if attention_mask is None and input_ids is not None:
        attention_mask = torch.ones_like(input_ids)
    return _forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


model.forward = _forward_with_default_mask
export_hf_checkpoint(model, export_dir=str(partial))
model.forward = _forward
print(f"exported to {partial} ({time.time() - t0:.0f}s)", flush=True)

export_dir = OutputDir.create(partial)
for rel, member in src_files.items():
    if not (partial / rel).exists():
        export_dir.copy(rel, member)
        print(f"  copied missing source file {rel}", flush=True)

cfg_src = src_config
cfg_exp = json.loads((partial / "config.json").read_text())
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("QUANT-FAILED: export config.json has no quantization_config")
# vLLM 0.32 serves Mistral3Model as an embedding model through its pooling adapter, which keeps the text tower's
# language-model head as a placeholder (`language_model.lm_head`, never loaded). Not on the exclusion list, it would get
# an NVFP4 quant method and fail at load ("Tensor.item() cannot be called on meta tensors", as in #32).
hq_path = partial / "hf_quant_config.json"
hq = json.loads(hq_path.read_text())
VLLM_HEADS = ("lm_head", "language_model.lm_head")
for container, key in ((hq["quantization"], "exclude_modules"), (qc, "ignore")):
    names = container.setdefault(key, [])
    for head in VLLM_HEADS:
        if head not in names:
            names.append(head)
    # ModelOpt writes exclusions under this checkpoint's names (language_model.layers.N.self_attn*), but vLLM 0.32 runs
    # the text tower as language_model.model.* and its Mistral3 prefix mapper only rewrites names that start with
    # model.language_model. A BF16 module excluded only under the checkpoint name is NOT excluded in vLLM: candidate C's
    # BF16 attention then loads as NVFP4 and fails (AssertionError in load_qkv_weight). Add the vLLM-internal twin of
    # every text-tower exclusion; the checkpoint-name entries stay for other loaders.
    for n in list(names):
        if n.startswith("language_model.") and not n.startswith(("language_model.model.", "language_model.lm_head")):
            twin = "language_model.model." + n[len("language_model."):]
            if twin not in names:
                names.append(twin)
hq_path.write_text(json.dumps(hq, indent=4) + "\n")
export_dir.copy("config.transformers5.json", CheckpointTree(partial).open("config.json"))
(partial / "config.json").write_text(json.dumps({**cfg_src, "quantization_config": qc}, indent=2) + "\n")
print(f"  restored source config.json (+ quantization_config, quant_algo {qc.get('quant_algo')}); "
      f"exclusions +{list(VLLM_HEADS)} for vLLM's pooling adapter", flush=True)
partial.rename(out)
out.with_name(out.name + ".ref.json").write_text(json.dumps(ref))
print(f"QUANT-OK {out} ({time.time() - t0:.0f}s)", flush=True)
