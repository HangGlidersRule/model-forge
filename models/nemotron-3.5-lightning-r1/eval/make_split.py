#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
The index and companion files must be regular files: a symlink aborts before OUT_DIR is touched, never followed.
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

_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
INDEX = "model.safetensors.index.json"


def bind_regular_files(d: Path) -> dict[str, tuple[int, os.stat_result]]:
    """Open the index and every companion file of d once; later reads use these descriptors, not the paths."""
    dir_fd = os.open(d, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    bound = {}
    try:
        for name in sorted(os.listdir(dir_fd)):
            if name.endswith(".safetensors"):
                continue
            try:
                fd = os.open(name, _READ, dir_fd=dir_fd)
            except OSError as e:
                if e.errno == errno.ELOOP:
                    raise SystemExit(f"{d / name} is a symlink; refusing to copy it") from None
                raise
            st = os.fstat(fd)
            if stat.S_ISREG(st.st_mode):
                bound[name] = (fd, st)
                continue
            os.close(fd)
            if not stat.S_ISDIR(st.st_mode):
                raise SystemExit(f"{d / name} is not a regular file")
    finally:
        os.close(dir_fd)
    if INDEX not in bound:
        raise SystemExit(f"{d / INDEX} is missing")
    return bound


def copy_into(out_fd: int, name: str, fd: int, st: os.stat_result) -> None:
    """copy2 from an open descriptor; the destination entry is replaced, never written through."""
    tmp = f".{name}.{os.getpid()}.tmp"
    dst = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=out_fd)
    try:
        with os.fdopen(fd, "rb") as fsrc, os.fdopen(dst, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, 1 << 24)
            os.fchmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
            fdst.flush()
            os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, name, src_dir_fd=out_fd, dst_dir_fd=out_fd)
    except BaseException:
        os.unlink(tmp, dir_fd=out_fd)
        raise


exp, out = Path(sys.argv[1]), Path(sys.argv[2])
bound = bind_regular_files(exp)
index_fd, _ = bound.pop(INDEX)
with os.fdopen(index_fd, "rb") as f:
    idx = json.loads(f.read())
out.mkdir(parents=True, exist_ok=True)
wm = idx["weight_map"]
new_map, scales = {}, {}
for f in sorted(set(wm.values())):
    tensors = {}
    with safe_open(exp / f, "pt") as h:
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
out_fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
for name, (fd, st) in bound.items():
    copy_into(out_fd, name, fd, st)
os.close(out_fd)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
