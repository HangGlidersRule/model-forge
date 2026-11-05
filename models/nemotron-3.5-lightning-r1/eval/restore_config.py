#!/usr/bin/env python3
"""Issue #17: copy only config.json + generation_config.json from the source checkpoint into a ModelOpt export, keeping
the export's quantization_config. transformers 5 re-serialises the config on export: it renames the layer-type
vocabulary (mamba -> linear_attention, attention -> full_attention), drops num_hidden_layers and adds
time_step_limit. vLLM reads the upstream vocabulary, so the served config must match the source's.
Writes the original export files as *.transformers5.json next to them for the record.
Usage: restore_config.py SOURCE_DIR EXPORT_DIR"""
import json
import os
import stat
import sys
from pathlib import Path


def read_checkpoint_file(path: Path, dir_fd: int) -> bytes:
    """Read ``path``, which must be a regular file and the entry ``path.name`` of the directory open as ``dir_fd``.

    A link is refused, including one swapped in after the directory was listed, and so is a file reached through a
    parent that was swapped for a link: the opened inode must be the one that directory lists under that name."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        raise SystemExit(f"{path}: {e.strerror} (checkpoint files must be regular files, not links)") from e
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or not os.path.samestat(
            st, os.stat(path.name, dir_fd=dir_fd, follow_symlinks=False)
        ):
            raise SystemExit(f"{path}: not a regular file in {path.parent}")
        with open(fd, "rb", closefd=False) as f:
            return f.read()
    finally:
        os.close(fd)


def write_output_file(path: Path, data: bytes) -> None:
    with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644), "wb") as f:
        f.write(data)


src, exp = Path(sys.argv[1]), Path(sys.argv[2])
src_fd = os.open(src, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
exp_fd = os.open(exp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
cfg_src = json.loads(read_checkpoint_file(src / "config.json", src_fd))
gen_src = read_checkpoint_file(src / "generation_config.json", src_fd)
exported = {name: read_checkpoint_file(exp / f"{name}.json", exp_fd) for name in ("config", "generation_config")}
cfg_exp = json.loads(exported["config"])
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("export config.json has no quantization_config")
# Idempotent: keep the first transformers-5 copies; a re-run must not overwrite them with the restored files.
for name, data in exported.items():
    if not os.path.lexists(exp / f"{name}.transformers5.json"):
        write_output_file(exp / f"{name}.transformers5.json", data)
cfg = dict(cfg_src)
cfg["quantization_config"] = qc
write_output_file(exp / "config.json", (json.dumps(cfg, indent=2) + "\n").encode())
write_output_file(exp / "generation_config.json", gen_src)
diff = sorted(k for k in set(cfg) | set(cfg_src) if cfg.get(k) != cfg_src.get(k))
print(json.dumps({"export": exp.name, "config_keys_differing_from_source": diff, "quant_algo": qc.get("quant_algo")}))
