#!/usr/bin/env python3
"""Issue #17: copy only config.json + generation_config.json from the source checkpoint into a ModelOpt export, keeping
the export's quantization_config. transformers 5 re-serialises the config on export: it renames the layer-type
vocabulary (mamba -> linear_attention, attention -> full_attention), drops num_hidden_layers and adds
time_step_limit. vLLM reads the upstream vocabulary, so the served config must match the source's.
Writes the original export files as *.transformers5.json next to them for the record.
Usage: restore_config.py SOURCE_DIR EXPORT_DIR"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from model_forge.pipeline import CheckpointTree, OutputDir  # noqa: E402

src, exp = Path(sys.argv[1]), Path(sys.argv[2])
# Every input is opened (no symlinks, regular files only) before anything is written.
src_tree, exp_tree = CheckpointTree(src), CheckpointTree(exp)
cfg_src = json.loads(src_tree.open("config.json").read_bytes())
gen_src = src_tree.open("generation_config.json")
exp_files = {name: exp_tree.open(f"{name}.json") for name in ("config", "generation_config")}
# Idempotent: keep the first transformers-5 copies; a re-run must not overwrite them with the restored files.
first_run = [name for name in exp_files if exp_tree.open_optional(f"{name}.transformers5.json") is None]
cfg_exp = json.loads(exp_files["config"].read_bytes())
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("export config.json has no quantization_config")
dst = OutputDir(exp_tree.fd, str(exp))
for name in first_run:
    dst.copy(f"{name}.transformers5.json", exp_files[name])
cfg = dict(cfg_src)
cfg["quantization_config"] = qc
dst.write_text("config.json", json.dumps(cfg, indent=2) + "\n")
dst.copy("generation_config.json", gen_src)
diff = sorted(k for k in set(cfg) | set(cfg_src) if cfg.get(k) != cfg_src.get(k))
print(json.dumps({"export": exp.name, "config_keys_differing_from_source": diff, "quant_algo": qc.get("quant_algo")}))
