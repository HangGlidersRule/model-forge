"""Path confinement for the Nano-Omni repair scripts (issue #35). Stdlib only, so the scripts that import it keep running
in the stock vLLM image without model_forge installed.

- shard_name(): the rule of model_forge.pipeline.shard_name(): an index weight_map value must be one plain file name.
- weight_map(): an index's weight_map, read through source_file(), with every shard name checked by shard_name().
- source_file(): the file a checkpoint directory holds at a relative path, which must resolve (after following any
  symlink) to a regular file inside that directory, or inside the Hub cache's blobs/ directory when the directory is a
  Hub cache snapshot (<repo>/snapshots/<revision>/, whose files are links to ../../blobs/<sha256>). Anything else is
  refused, so a link or a crafted name cannot pull bytes from elsewhere into a published tree. The check is on the
  resolved path, so it needs no lexical rules of its own: `..`, absolute names and links all resolve before it.
"""
from __future__ import annotations

import json
from pathlib import Path


class CheckpointPathError(ValueError):
    pass


def shard_name(name: object) -> str:
    """Return ``name`` if it is a single file name with no directory part."""
    if not isinstance(name, str) or name in ("", ".", "..") or any(sep in name for sep in ("/", "\\", "\x00")):
        raise CheckpointPathError(f"shard name must be a single file name, got {name!r}")
    return name


def _allowed_roots(real_root: Path) -> tuple[Path, ...]:
    if real_root.parent.name == "snapshots":
        return real_root, real_root.parent.parent / "blobs"
    return (real_root,)


def source_file(root: Path, rel: str | Path) -> Path:
    """The resolved regular file ``root / rel``, confined to ``root`` (and a Hub snapshot's blobs/)."""
    try:
        real_root = root.resolve(strict=True)
        target = (root / rel).resolve(strict=True)
    except (OSError, ValueError) as error:
        raise CheckpointPathError(f"{root}/{rel}: cannot resolve ({error})") from None
    if not any(target.is_relative_to(allowed) for allowed in _allowed_roots(real_root)):
        raise CheckpointPathError(f"{root}/{rel} resolves outside {root}")
    if not target.is_file():
        raise CheckpointPathError(f"{root}/{rel} is not a regular file")
    return target


def weight_map(root: Path) -> dict[str, str]:
    """The index's weight_map with every shard name checked."""
    index = json.loads(source_file(root, "model.safetensors.index.json").read_text())
    wm = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(wm, dict):
        raise CheckpointPathError(f"{root}: model.safetensors.index.json has no weight_map mapping")
    return {tensor: shard_name(shard) for tensor, shard in wm.items()}
