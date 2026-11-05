#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Every file read from the export must be a regular file: a symlink is refused, never followed.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import contextlib
import errno
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def open_regular(path: Path) -> int:
    """Open path without following a symlink at its last component; refuse anything but a regular file."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise SystemExit(f"refusing symlink in export: {path}") from None
        raise
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise SystemExit(f"refusing non-regular file in export: {path}")
    return fd


def copy_fd(fd: int, dst: Path) -> None:
    """copy2 from an open file: the bytes go to a new file beside dst that then replaces dst, never through a link."""
    st = os.fstat(fd)
    tmp_fd, tmp = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.")
    try:
        with open(tmp_fd, "wb") as w, open(fd, "rb", closefd=False) as r:
            r.seek(0)
            shutil.copyfileobj(r, w)
            w.flush()
            os.fchmod(w.fileno(), stat.S_IMODE(st.st_mode))
            os.utime(w.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))
        os.replace(tmp, dst)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def shard_name(f: str) -> str:
    if f in (".", "..") or Path(f).name != f:
        raise SystemExit(f"index names a shard outside the export: {f!r}")
    return f


exp, out = Path(sys.argv[1]), Path(sys.argv[2])
idx_fd = open_regular(exp / "model.safetensors.index.json")
with open(idx_fd, "rb") as h:
    idx = json.loads(h.read())
wm = idx["weight_map"]
shards = {f: open_regular(exp / shard_name(f)) for f in sorted(set(wm.values()))}
companions = {}
with os.scandir(exp) as it:
    for e in sorted(it, key=lambda e: e.name):
        if e.name.endswith(".safetensors") or e.name == "model.safetensors.index.json":
            continue
        if e.is_dir(follow_symlinks=False):
            continue
        companions[e.name] = open_regular(exp / e.name)

out.mkdir(parents=True, exist_ok=True)
new_map, scales = {}, {}
for f, fd in shards.items():
    tensors = {}
    with safe_open(f"/proc/self/fd/{fd}", "pt") as h:
        meta = h.metadata() or {"format": "pt"}
        for k in h.keys():
            (scales if k.endswith(".input_scale") else tensors)[k] = h.get_tensor(k)
    os.close(fd)
    save_file(tensors, out / f, metadata=meta)
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
save_file(scales, out / "model-input-scales.safetensors", metadata={"format": "pt"})
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
(out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")
for name, fd in companions.items():
    copy_fd(fd, out / name)
    os.close(fd)
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
