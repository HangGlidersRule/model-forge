#!/usr/bin/env python3
"""Issue #17: copy only config.json + generation_config.json from the source checkpoint into a ModelOpt export, keeping
the export's quantization_config. transformers 5 re-serialises the config on export: it renames the layer-type
vocabulary (mamba -> linear_attention, attention -> full_attention), drops num_hidden_layers and adds
time_step_limit. vLLM reads the upstream vocabulary, so the served config must match the source's.
Writes the original export files as *.transformers5.json next to them for the record.
Usage: restore_config.py SOURCE_DIR EXPORT_DIR"""
import json
import shutil
import sys
from pathlib import Path

src, exp = Path(sys.argv[1]), Path(sys.argv[2])
cfg_src = json.loads((src / "config.json").read_text())
cfg_exp = json.loads((exp / "config.json").read_text())
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("export config.json has no quantization_config")
# Idempotent: keep the first transformers-5 copies; a re-run must not overwrite them with the restored files.
for name in ("config", "generation_config"):
    if not (exp / f"{name}.transformers5.json").exists():
        shutil.copyfile(exp / f"{name}.json", exp / f"{name}.transformers5.json")
cfg = dict(cfg_src)
cfg["quantization_config"] = qc
(exp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
shutil.copyfile(src / "generation_config.json", exp / "generation_config.json")
diff = sorted(k for k in set(cfg) | set(cfg_src) if cfg.get(k) != cfg_src.get(k))
print(json.dumps({"export": exp.name, "config_keys_differing_from_source": diff, "quant_algo": qc.get("quant_algo")}))
