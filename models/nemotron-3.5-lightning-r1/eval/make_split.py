#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Every export file used is opened once, without following symlinks, before OUT_DIR is touched; a symlink stops the run.
Shards are read, and non-weight files copied, through those descriptors.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import errno
import json
import os
import shutil
import stat
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def fd_path(fd):
    """Procfs path that opens, or links, exactly the inode FD holds. Built from parts because the public export rewrites
    absolute path literals."""
    return os.path.join(os.sep, "proc", "self", "fd", str(fd))


def open_entry(dfd, name, where):
    """Open NAME in directory DFD without following a symlink. Returns None for a directory or special file."""
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=dfd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink {where / name}: inputs must be regular files") from None
        raise
    if stat.S_ISREG(os.fstat(fd).st_mode):
        return fd
    os.close(fd)
    return None


def copy_bound(fd, dst):
    """Copy the file open as FD to DST with its mode and timestamps, as shutil.copy2 would."""
    st = os.fstat(fd)
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb", closefd=False) as r, open(dst, "wb") as w:
        shutil.copyfileobj(r, w)
    os.chmod(dst, stat.S_IMODE(st.st_mode))
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))


exp, out = Path(sys.argv[1]), Path(sys.argv[2])
exp_fd = os.open(exp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
names = set(os.listdir(exp_fd))
idx_fd = open_entry(exp_fd, "model.safetensors.index.json", exp) if "model.safetensors.index.json" in names else None
if idx_fd is None:
    raise SystemExit(f"{exp / 'model.safetensors.index.json'} is not a regular file")
with os.fdopen(idx_fd, "rb", closefd=False) as r:
    idx = json.loads(r.read())
wm = idx["weight_map"]
shard_fds = {}
for f in sorted(set(wm.values())):
    fd = open_entry(exp_fd, f, exp) if f in names else None
    if fd is None:
        raise SystemExit(f"shard {f} named by the index is not a regular file in {exp}")
    shard_fds[f] = fd
extra_fds = {}
for name in sorted(names):
    if not name.endswith(".safetensors") and name != "model.safetensors.index.json":
        fd = open_entry(exp_fd, name, exp)
        if fd is not None:
            extra_fds[name] = fd

out.mkdir(parents=True, exist_ok=True)
new_map, scales = {}, {}
for f, fd in shard_fds.items():
    tensors = {}
    with safe_open(fd_path(fd), "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            (scales if k.endswith(".input_scale") else tensors)[k] = h.get_tensor(k)
    save_file(tensors, out / f, metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
save_file(scales, out / "model-input-scales.safetensors", metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
(out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
for name, fd in extra_fds.items():
    copy_bound(fd, out / name)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
