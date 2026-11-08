#!/usr/bin/env python3
"""Issue #17 R17-A1: re-lay candidate A's export so every activation input_scale tensor lives in one small shard
(model-input-scales.safetensors). The big shards are rewritten once without them; each scale variant then hardlinks
the big shards and carries only its own small shard. Tensor values are unchanged.
Usage: make_split.py EXPORT_DIR OUT_DIR"""
import json
import os
import stat
import sys
from pathlib import Path

from safetensors.torch import load, save

from model_forge.pipeline import (
    INDEX_NAME,
    OutputDir,
    SourceDir,
    safetensors_metadata,
    shard_names,
)


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


exp, out = Path(sys.argv[1]), Path(sys.argv[2])
exp_fd = os.open(exp, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
idx = json.loads(read_checkpoint_file(exp / INDEX_NAME, exp_fd))
wm = shard_names(idx["weight_map"])
sidecars = {
    name: read_checkpoint_file(exp / name, exp_fd)
    for name in sorted(os.listdir(exp_fd))
    if not name.endswith(".safetensors") and name != INDEX_NAME
    and not stat.S_ISDIR(os.stat(name, dir_fd=exp_fd, follow_symlinks=False).st_mode)
}
os.close(exp_fd)
src = SourceDir(exp)
shards = {f: src.open(f) for f in sorted(set(wm.values()))}
out.mkdir(parents=True, exist_ok=True)
dst = OutputDir(out)
new_map, scales = {}, {}
for f, shard in shards.items():
    data = shard.read_bytes()
    meta = safetensors_metadata(data) or {"format": "pt"}
    tensors = {}
    for k, t in load(data).items():
        (scales if k.endswith(".input_scale") else tensors)[k] = t
    del data
    dst.write_bytes(f, save(tensors, metadata=meta))
    new_map.update({k: f for k in tensors})
    print(f, len(tensors), "tensors", flush=True)
dst.write_bytes("model-input-scales.safetensors", save(scales, metadata={"format": "pt"}))
new_map.update({k: "model-input-scales.safetensors" for k in scales})
idx["weight_map"] = dict(sorted(new_map.items()))
dst.write_bytes(INDEX_NAME, (json.dumps(idx, indent=2) + "\n").encode())
for name, data in sidecars.items():
    dst.write_bytes(name, data)
src.close()
assert len(new_map) == len(wm), (len(new_map), len(wm))
print(json.dumps({"input_scales": len(scales), "tensors_total": len(new_map)}))
