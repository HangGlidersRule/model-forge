#!/usr/bin/env python3
"""Apply abliteration projection shard-by-shard to official Qwen3.8-27B BF16.

Processes safetensor shards without holding two full model copies.
Applies W' = W - r(r^T W) in float32 to exactly the expected 131 tensors.
Copies all other tensors byte-for-byte. Preserves tokenizer, processor, and config.
Non-safetensor assets must be regular files. The only symlinks followed are a Hugging Face cache snapshot's links into
its own real blobs directory, and from there a per-repo blob link of the form ../../blobs/<prefix>/<xet_hash> to a
regular file in the cache-wide shared store. Any other symlink aborts the run before anything is written.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import posixpath
import shutil
import stat
import sys
from pathlib import Path, PurePosixPath

import torch

from model_forge.abliteration import (
    compute_leakage,
    is_vision_tensor,
    matches_selector,
    project_weight,
)
from model_forge.experiment import load_experiment
from model_forge.pipeline import (
    RunLock,
    StageContext,
    SuccessManifest,
    sha256_file,
)

_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_REAL_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
Asset = tuple[PurePosixPath, int | None, os.stat_result | None]


def _is_symlink(path: str | Path, dir_fd: int | None = None) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path, dir_fd=dir_fd).st_mode)
    except OSError:
        return False


def _open_real_directory(path: str | Path, dir_fd: int | None = None) -> int:
    """Open path as a directory. A final-component symlink is refused rather than followed."""
    try:
        return os.open(path, _REAL_DIR, dir_fd=dir_fd)
    except OSError as e:
        if e.errno in (errno.ELOOP, errno.ENOTDIR) and _is_symlink(path, dir_fd):
            raise SystemExit(f"ERROR: {path} is a symlink; refusing to copy through it") from None
        raise


def open_hf_blobs(model_path: Path) -> int | None:
    """The real blobs directory of a Hugging Face cache snapshot (<repo>/snapshots/<rev>), or None."""
    if model_path.parent.name != "snapshots":
        return None
    try:
        return _open_real_directory(model_path.parent.parent / "blobs")
    except FileNotFoundError:
        return None


def _shared_blob_parts(link: str) -> tuple[str, str] | None:
    """Split a huggingface_hub>=1.32 per-repo blob link, ../../blobs/<prefix>/<xet_hash>."""
    parent, name = posixpath.split(link)
    root, prefix = posixpath.split(parent)
    if root != "../../blobs" or len(prefix) != 2 or len(name) != 64 or prefix != name[:2]:
        return None
    if any(c not in "0123456789abcdef" for c in name):
        return None
    return prefix, name


def _open_shared_blob(blobs_fd: int, link: str) -> int:
    """Open the regular file a per-repo blob symlink names in the cache-wide shared store.

    ``link`` is resolved from the already-open real blobs directory, and only the fixed relative shape
    ``../../blobs/<prefix>/<xet_hash>``. ``blobs``, the prefix directory, and the file are opened with
    ``O_NOFOLLOW``, so a symlink anywhere in that chain is refused.
    """
    parts = _shared_blob_parts(link)
    if parts is None:
        raise SystemExit(
            f"ERROR: blob symlink {link!r} is outside the Hugging Face shared blob store; not copied"
        )
    prefix, name = parts
    repo_fd = _open_real_directory("..", dir_fd=blobs_fd)
    try:
        cache_fd = _open_real_directory("..", dir_fd=repo_fd)
        try:
            store_fd = _open_real_directory("blobs", dir_fd=cache_fd)
            try:
                prefix_fd = _open_real_directory(prefix, dir_fd=store_fd)
            finally:
                os.close(store_fd)
        finally:
            os.close(cache_fd)
    finally:
        os.close(repo_fd)
    try:
        try:
            fd = os.open(name, _READ, dir_fd=prefix_fd)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise SystemExit(f"ERROR: shared blob {name} is a symlink; refusing to copy it") from None
            raise
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise SystemExit(f"ERROR: shared blob {name} is not a regular file; refusing to copy it")
        return fd
    finally:
        os.close(prefix_fd)


def open_asset(model_path: Path, dir_fd: int, rel: PurePosixPath, blobs_fd: int | None) -> int:
    """Open rel without following arbitrary symlinks. A relative link that names an entry directly inside the cache's
    real blobs directory is opened there. A per-repo blob that links to ../../blobs/<prefix>/<xet_hash> is opened as
    that regular file. Any other symlink aborts."""
    try:
        return os.open(rel.name, _READ, dir_fd=dir_fd)
    except OSError as e:
        if e.errno != errno.ELOOP:
            raise
    target = os.readlink(rel.name, dir_fd=dir_fd)
    head, blob = posixpath.split(posixpath.normpath(posixpath.join("snapshots", "rev", str(rel.parent), target)))
    if blobs_fd is None or posixpath.isabs(target) or head != "blobs" or blob in ("", ".", ".."):
        raise SystemExit(f"ERROR: {model_path / rel} is a symlink outside the Hugging Face blob store; not copied")
    try:
        return os.open(blob, _READ, dir_fd=blobs_fd)
    except OSError as e:
        if e.errno != errno.ELOOP:
            raise
    return _open_shared_blob(blobs_fd, os.readlink(blob, dir_fd=blobs_fd))


def bind_assets(model_path: Path, blobs_fd: int | None) -> list[Asset]:
    """Open every non-safetensor asset once, before any output exists: (rel, fd, stat) per file, (rel, None, None) per
    directory, parents first. The copy later reads these descriptors, not the paths."""
    bound: list[Asset] = []

    def walk(dir_fd: int, rel_dir: PurePosixPath, top: bool) -> None:
        for name in sorted(os.listdir(dir_fd)):
            if top and (name.endswith(".safetensors") or name.startswith(".")):
                continue
            rel = rel_dir / name
            fd = open_asset(model_path, dir_fd, rel, blobs_fd)
            st = os.fstat(fd)
            if stat.S_ISREG(st.st_mode):
                bound.append((rel, fd, st))
            elif stat.S_ISDIR(st.st_mode):
                bound.append((rel, None, None))
                walk(fd, rel, top=False)
                os.close(fd)
            else:
                raise SystemExit(f"ERROR: {model_path / rel} is not a regular file or directory")

    root_fd = os.open(model_path, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    walk(root_fd, PurePosixPath("."), top=True)
    os.close(root_fd)
    return bound


def copy_assets(bound: list[Asset], dest_root: Path) -> None:
    """shutil.copy2 / copytree from the bound descriptors into fresh entries under dest_root."""
    for rel, fd, st in bound:
        dest = dest_root / rel
        if fd is None or st is None:
            dest.mkdir()
            continue
        out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "rb") as fsrc, os.fdopen(out, "wb") as fdst:
            shutil.copyfileobj(fsrc, fdst, 1 << 24)
            os.fchmod(fdst.fileno(), stat.S_IMODE(st.st_mode))
            fdst.flush()
            os.utime(fdst.fileno(), ns=(st.st_atime_ns, st.st_mtime_ns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply abliteration to BF16 model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--direction-dir", type=Path, default=None)
    parser.add_argument("--source-model", type=Path, default=None, help="Local model path (optional)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-stage", action="store_true")
    args = parser.parse_args()

    cfg = load_experiment(args.config)
    cfg_sha = cfg.config_sha()

    ctx = StageContext(args.run_root, "apply_abliteration", cfg_sha)
    if args.resume and not args.force_stage:
        existing = ctx.should_skip()
        if existing:
            print(f"Abliteration stage verified, skipping (config_sha={cfg_sha[:12]})")
            return

    direction_dir = args.direction_dir or (args.run_root / "measure_direction")
    direction_file = direction_dir / "direction.safetensors"
    if not direction_file.exists():
        print("ERROR: Direction not measured. Run measure_qwen38_refusal_direction.py first.", file=sys.stderr)
        sys.exit(1)

    from safetensors.torch import load_file, save_file

    direction_tensors = load_file(str(direction_file))
    direction = direction_tensors["direction"]

    if args.source_model:
        model_path = args.source_model
    else:
        from huggingface_hub import snapshot_download

        print(f"Downloading {cfg.source.model_id}@{cfg.source.revision[:12]}...")
        model_path = Path(snapshot_download(cfg.source.model_id, revision=cfg.source.revision))

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        print("ERROR: model.safetensors.index.json not found", file=sys.stderr)
        sys.exit(1)

    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    selectors = list(cfg.abliteration.target_selectors)
    target_names = [name for name in weight_map if matches_selector(name, selectors)]

    for name in target_names:
        if is_vision_tensor(name):
            print(f"ERROR: Selector matched vision tensor: {name}", file=sys.stderr)
            sys.exit(1)

    if len(target_names) != cfg.abliteration.expected_target_count:
        print(
            f"ERROR: Expected {cfg.abliteration.expected_target_count} targets, "
            f"found {len(target_names)}: {sorted(target_names)[:5]}...",
            file=sys.stderr,
        )
        sys.exit(1)

    assets = bind_assets(model_path, open_hf_blobs(model_path))

    print(f"Applying projection to {len(target_names)} tensors...")
    lock = RunLock(args.run_root / "apply_abliteration.lock")
    with lock:
        partial = ctx.partial_dir
        target_set = set(target_names)
        shard_files = sorted(set(weight_map.values()))

        edit_report: list[dict[str, object]] = []
        edited_count = 0
        unchanged_vision_count = 0

        for shard_name in shard_files:
            src_shard = model_path / shard_name
            shard_tensors = load_file(str(src_shard))
            modified_tensors: dict[str, torch.Tensor] = {}

            for tname, tensor in shard_tensors.items():
                if tname in target_set:
                    before_norm = tensor.float().norm().item()
                    projected = project_weight(tensor, direction)
                    after_norm = projected.float().norm().item()
                    leakage = compute_leakage(projected, direction)
                    modified_tensors[tname] = projected
                    edit_report.append({
                        "tensor": tname,
                        "before_norm": before_norm,
                        "after_norm": after_norm,
                        "leakage": leakage,
                        "edited": True,
                    })
                    edited_count += 1
                else:
                    modified_tensors[tname] = tensor
                    if is_vision_tensor(tname):
                        unchanged_vision_count += 1

            save_file(modified_tensors, str(partial / shard_name))

        assert edited_count == cfg.abliteration.expected_target_count

        copy_assets(assets, partial)

        # Write edit report
        report_path = partial / "abliteration_report.json"
        report_path.write_text(json.dumps({
            "edited_count": edited_count,
            "expected_count": cfg.abliteration.expected_target_count,
            "edits": edit_report,
            "unchanged_vision_count": unchanged_vision_count,
        }, indent=2) + "\n")

        # Compute output hashes for key files
        output_hashes: dict[str, str] = {}
        for f in sorted(partial.rglob("*")):
            if f.is_file() and f.name != "_SUCCESS.json":
                output_hashes[str(f.relative_to(partial))] = sha256_file(f)

        manifest = SuccessManifest(
            stage="apply_abliteration",
            config_sha=cfg_sha,
            source_revisions={"model": cfg.source.revision},
            output_hashes=output_hashes,
            metrics={
                "edited_tensors": edited_count,
                "max_leakage": max((e["leakage"] for e in edit_report), default=0.0),
            },
        )
        ctx.promote(manifest)
        print(f"Abliteration applied: {edited_count} tensors edited")


if __name__ == "__main__":
    main()
