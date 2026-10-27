#!/usr/bin/env python3
"""Verify merge_lora.py against PEFT's own merge_and_unload on a tiny random Qwen3.5 model.

Builds a 2-layer Qwen3_5ForCausalLM (text) with random weights, attaches a random LoRA with the
JEV adapter's exact config (r=32, alpha=64, same target_modules incl. lm_head), then:
  1. PEFT merge_and_unload -> reference weights
  2. merge_lora.py on the saved base + adapter (keys renamed into JEV's published layout and the
     base into the multimodal 'model.language_model.' layout the 27B base uses)
and asserts every tensor matches. Prints PEFT-EQUIVALENT on success.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

torch.manual_seed(1234)
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoConfig, AutoModelForCausalLM  # noqa: E402

tmp = Path(tempfile.mkdtemp())
cfg = AutoConfig.for_model(
    "qwen3_5_text", hidden_size=256, intermediate_size=512, num_hidden_layers=4, num_attention_heads=4,
    num_key_value_heads=2, head_dim=64, vocab_size=1024, full_attention_interval=4,
    linear_num_key_heads=4, linear_num_value_heads=8, linear_key_head_dim=32, linear_value_head_dim=32,
    tie_word_embeddings=False, dtype="bfloat16",
)
model = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
base_sd = {k: v.clone() for k, v in model.state_dict().items()}
targets = ["down_proj", "gate_proj", "in_proj_qkv", "in_proj_z", "k_proj", "lm_head", "o_proj", "out_proj", "q_proj", "up_proj", "v_proj"]
lcfg = LoraConfig(r=32, lora_alpha=64, target_modules=targets, lora_dropout=0.0, bias="none", init_lora_weights=False)
peft_model = get_peft_model(model, lcfg)
# Fixtures must match the published JEV adapter: LoRA A/B tensors only. PEFT's lm_head target saves the
# base lm_head as well, which JEV's adapter_vllm does not ship.
peft_model.save_pretrained(tmp / "adapter", save_embedding_layers=False)
_ad = load_file(str(tmp / "adapter" / "adapter_model.safetensors"))
_lora = {k: v for k, v in _ad.items() if ".lora_A." in k or ".lora_B." in k}
assert len(_lora) == len(_ad) - sum(1 for k in _ad if "base_layer" in k), sorted(set(_ad) - set(_lora))
save_file(_lora, str(tmp / "adapter" / "adapter_model.safetensors"))
ref = peft_model.merge_and_unload().state_dict()

# Base in the 27B checkpoint's multimodal layout.
mm = {("lm_head.weight" if k == "lm_head.weight" else k.replace("model.", "model.language_model.", 1)): v for k, v in base_sd.items()}
(tmp / "base").mkdir()
save_file({k: v.contiguous() for k, v in mm.items()}, str(tmp / "base" / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
(tmp / "base" / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in mm}}))
(tmp / "base" / "config.json").write_text("{}")
# PEFT already saves the published layout (base_model.model.model.layers.N...); add a decision_head stub.
(tmp / "adapter" / "decision_head.json").write_text("{}")
akeys = list(load_file(str(tmp / "adapter" / "adapter_model.safetensors")))
print("adapter tensors:", len(akeys), "| sample:", akeys[:2])

subprocess.run([sys.executable, str(Path(__file__).with_name("merge_lora.py")), "--base", str(tmp / "base"),
                "--adapter", str(tmp / "adapter"), "--out", str(tmp / "out")], check=True)
out = load_file(str(tmp / "out" / "model-00001-of-00001.safetensors"))
bad, worst = [], 0.0
for k, v in ref.items():
    ok_key = "lm_head.weight" if k == "lm_head.weight" else k.replace("model.", "model.language_model.", 1)
    o = out[ok_key]
    d = (o.float() - v.float()).abs().max().item()
    worst = max(worst, d)
    # Both round to BF16 once from an FP32 sum; allow 1 BF16 ulp where PEFT accumulates in BF16.
    if d > 0 and not torch.allclose(o.float(), v.float(), rtol=8e-3, atol=1e-3):
        bad.append((k, d))
changed = sum(1 for k in ref if not torch.equal(ref[k], base_sd[k]))
print(f"tensors {len(ref)} | changed by merge {changed} | max abs diff vs PEFT {worst:.3e} | outside tolerance {len(bad)}")
print("PEFT-EQUIVALENT" if not bad and changed > 0 else f"MISMATCH {bad[:5]}")
if len(sys.argv) > 1:
    # Optional evidence record: the measured comparison, not just the pass/fail verdict.
    import peft
    import transformers
    Path(sys.argv[1]).write_text(json.dumps({
        "check": "merge_lora.py vs PEFT merge_and_unload on a reduced random Qwen3.5 text model",
        "model": {"layers": 4, "hidden_size": 256, "vocab_size": 1024, "dtype": "bfloat16", "seed": 1234},
        "lora": {"r": 32, "lora_alpha": 64, "target_modules": targets},
        "tensors": len(ref), "changed_by_merge": changed, "max_abs_diff_vs_peft": worst,
        "bit_identical": worst == 0.0, "outside_tolerance": len(bad),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }, indent=1) + "\n")
sys.exit(0 if not bad and changed > 0 else 1)
