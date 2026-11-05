#!/usr/bin/env python3
"""Issue #17: copy only config.json + generation_config.json from the source checkpoint into a ModelOpt export, keeping
the export's quantization_config. transformers 5 re-serialises the config on export: it renames the layer-type
vocabulary (mamba -> linear_attention, attention -> full_attention), drops num_hidden_layers and adds
time_step_limit. vLLM reads the upstream vocabulary, so the served config must match the source's.
Writes the original export files as *.transformers5.json next to them for the record.
Every file read must be a regular file (a symlink is refused, never followed), and every file written replaces the
path in the export rather than writing through whatever is there.
Usage: restore_config.py SOURCE_DIR EXPORT_DIR"""
import contextlib
import errno
import json
import os
import stat
import sys
import tempfile
from pathlib import Path


def read_regular(path: Path) -> bytes:
    """Read path without following a symlink at its last component; refuse anything but a regular file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink: {path}") from None
        raise
    with open(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SystemExit(f"refusing non-regular file: {path}")
        return f.read()


def replace_with(dst: Path, data: bytes) -> None:
    tmp_fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    try:
        with open(tmp_fd, "wb") as w:
            w.write(data)
            os.fchmod(w.fileno(), 0o644)
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


src, exp = Path(sys.argv[1]), Path(sys.argv[2])
cfg_src = json.loads(read_regular(src / "config.json"))
gen_src = read_regular(src / "generation_config.json")
cfg_exp_raw = read_regular(exp / "config.json")
cfg_exp = json.loads(cfg_exp_raw)
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("export config.json has no quantization_config")
# Idempotent: keep the first transformers-5 copies; a re-run must not overwrite them with the restored files.
originals = {}
for name in ("config", "generation_config"):
    if not (exp / f"{name}.transformers5.json").exists():
        originals[name] = cfg_exp_raw if name == "config" else read_regular(exp / f"{name}.json")
for name, data in originals.items():
    replace_with(exp / f"{name}.transformers5.json", data)
cfg = dict(cfg_src)
cfg["quantization_config"] = qc
replace_with(exp / "config.json", (json.dumps(cfg, indent=2) + "\n").encode())
replace_with(exp / "generation_config.json", gen_src)
diff = sorted(k for k in set(cfg) | set(cfg_src) if cfg.get(k) != cfg_src.get(k))
print(json.dumps({"export": exp.name, "config_keys_differing_from_source": diff, "quant_algo": qc.get("quant_algo")}))
