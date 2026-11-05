#!/usr/bin/env python3
"""Issue #17: copy only config.json + generation_config.json from the source checkpoint into a ModelOpt export, keeping
the export's quantization_config. transformers 5 re-serialises the config on export: it renames the layer-type
vocabulary (mamba -> linear_attention, attention -> full_attention), drops num_hidden_layers and adds
time_step_limit. vLLM reads the upstream vocabulary, so the served config must match the source's.
Writes the original export files as *.transformers5.json next to them for the record.
Every file read or replaced must be a regular file: a symlink is refused, never followed, and nothing in the export is
written until all of them have been opened and checked.
Usage: restore_config.py SOURCE_DIR EXPORT_DIR"""
import errno
import json
import os
import stat
import sys
from pathlib import Path

_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def open_dir(path: Path) -> int:
    return os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def read_regular(dir_fd: int, name: str, where: Path) -> tuple[bytes, int] | None:
    """(bytes, permission bits) of a regular file in dir_fd, or None if absent. Symlinks and special files abort."""
    try:
        fd = os.open(name, _READ, dir_fd=dir_fd)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"{where / name} is a symlink; refusing to follow it") from None
        raise
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise SystemExit(f"{where / name} is not a regular file")
        return f.read(), stat.S_IMODE(st.st_mode)


def require_regular(dir_fd: int, name: str, where: Path) -> tuple[bytes, int]:
    found = read_regular(dir_fd, name, where)
    if found is None:
        raise SystemExit(f"{where / name} is missing")
    return found


def write_new(dir_fd: int, name: str, data: bytes, mode: int) -> None:
    fd = os.open(name, _CREATE, 0o600, dir_fd=dir_fd)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            os.fchmod(f.fileno(), mode)
    except BaseException:
        os.unlink(name, dir_fd=dir_fd)
        raise


def replace(dir_fd: int, name: str, data: bytes, mode: int) -> None:
    """Swap name for a new regular file; whatever sits at name is replaced, never written through."""
    tmp = f".{name}.{os.getpid()}.tmp"
    write_new(dir_fd, tmp, data, mode)
    try:
        os.replace(tmp, name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except BaseException:
        os.unlink(tmp, dir_fd=dir_fd)
        raise


src, exp = Path(sys.argv[1]), Path(sys.argv[2])
src_fd, exp_fd = open_dir(src), open_dir(exp)
cfg_src_raw, _ = require_regular(src_fd, "config.json", src)
gen_src_raw, _ = require_regular(src_fd, "generation_config.json", src)
originals = {name: require_regular(exp_fd, f"{name}.json", exp) for name in ("config", "generation_config")}
cfg_src = json.loads(cfg_src_raw)
cfg_exp = json.loads(originals["config"][0])
qc = cfg_exp.get("quantization_config")
if not qc:
    raise SystemExit("export config.json has no quantization_config")
# Idempotent: keep the first transformers-5 copies; a re-run must not overwrite them with the restored files.
pending = [name for name in originals if read_regular(exp_fd, f"{name}.transformers5.json", exp) is None]
for name in pending:
    write_new(exp_fd, f"{name}.transformers5.json", *originals[name])
cfg = dict(cfg_src)
cfg["quantization_config"] = qc
replace(exp_fd, "config.json", (json.dumps(cfg, indent=2) + "\n").encode(), originals["config"][1])
replace(exp_fd, "generation_config.json", gen_src_raw, originals["generation_config"][1])
diff = sorted(k for k in set(cfg) | set(cfg_src) if cfg.get(k) != cfg_src.get(k))
print(json.dumps({"export": exp.name, "config_keys_differing_from_source": diff, "quant_algo": qc.get("quant_algo")}))
