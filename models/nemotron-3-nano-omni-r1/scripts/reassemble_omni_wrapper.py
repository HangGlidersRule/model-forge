#!/usr/bin/env python3
"""Reassemble a Nano-Omni VLM checkpoint from a published trunk-only NVFP4 product and the upstream BF16 wrapper
(issue #35). Byte-level: no torch, no dequantization, no re-quantization.

Inputs:
  TRUNK     a published Darkstar Nano-Omni ModelOpt NVFP4 repository snapshot. The trunk was quantized as a standalone
            `nemotron_h` checkpoint: tensors named backbone.* and lm_head.*, config.json with a MIXED_PRECISION
            quantization_config over those names.
  UPSTREAM  nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 at the family's pinned revision: the wrapper config, the
            wrapper code and processor files, and the 1,106 vision, audio and projector tensors.
Output (OUT): a `NemotronH_Nano_Omni_Reasoning_V3` checkpoint:
  - every trunk tensor's bytes unchanged, renamed backbone.* -> language_model.backbone.* and lm_head.* ->
    language_model.lm_head.* (the upstream wrapper's names);
  - every non-language_model tensor of UPSTREAM (vision tower, audio encoder, projectors) with its bytes unchanged, in a
    separate shard;
  - config.json = upstream's wrapper config, verbatim, plus `quantization_config` = the trunk's, with every module name
    in `quantized_layers` (and in `ignore`, if any) given the same prefix;
  - hf_quant_config.json = the trunk's, renamed the same way;
  - every other file (code, processor, tokenizer, chat template, generation config) copied from UPSTREAM byte for byte;
  - model.safetensors.index.json over the union, and manifest.sha256 over every file.
Prints REASSEMBLE-OK with counts. Fails closed on any name collision, missing file or count mismatch, on an index
shard name that is not one plain file name, and on any input file that resolves outside its checkpoint directory
(a Hub cache snapshot's links into its blobs/ directory are allowed; checkpoint_paths.py).
Usage: reassemble_omni_wrapper.py TRUNK UPSTREAM OUT"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

# The confinement helper sits beside this script (stdlib only); put its directory on the path so the import works
# however the script is loaded (run directly in a container, or by file path from the tests).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_paths import CheckpointPathError, source_file, weight_map  # noqa: E402

LM = "language_model."
SHARD_TOWERS = "model-towers.safetensors"
# Files that come from the trunk export or are regenerated here; everything else is copied from UPSTREAM.
GENERATED = {"config.json", "hf_quant_config.json", "model.safetensors.index.json", "manifest.sha256"}
UPSTREAM_SKIP_SUFFIXES = (".safetensors",)
UPSTREAM_SKIP_NAMES = {"SHA256SUMS.upstream", "SOURCE_REVISION", "README.md", ".gitattributes"}


def lm_name(name: str) -> str:
    """Trunk tensor or module name -> the upstream wrapper's name for it."""
    if name.startswith(LM):
        raise ValueError(f"trunk name already carries the wrapper prefix: {name}")
    if not (name.startswith("backbone.") or name == "lm_head" or name.startswith("lm_head.")):
        raise ValueError(f"unexpected trunk name (not backbone.* or lm_head.*): {name}")
    return LM + name


def read_header(path: Path) -> tuple[dict, int]:
    """Return (header dict, offset of the data section) of a safetensors file."""
    with path.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def write_renamed(src: Path, dst: Path, rename) -> dict[str, str]:
    """Copy a safetensors file with renamed tensor keys. The data section is copied verbatim and every tensor keeps its
    dtype, shape and byte range; only the header changes. Returns {new_name: dst.name}."""
    header, data_off = read_header(src)
    meta = header.pop("__metadata__", None)
    new = {rename(k): v for k, v in header.items()}
    if len(new) != len(header):
        raise ValueError(f"{src.name}: renaming produced duplicate names")
    if meta is not None:
        new["__metadata__"] = meta
    blob = json.dumps(new, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        fin.seek(data_off)
        shutil.copyfileobj(fin, fout, 1 << 24)
    return {k: dst.name for k in new if k != "__metadata__"}


def write_subset(sources: dict[Path, list[str]], dst: Path) -> dict[str, str]:
    """Write the named tensors of several safetensors files into one new file, bytes unchanged."""
    entries, chunks, offset = {}, [], 0
    for src, names in sources.items():
        header, data_off = read_header(src)
        for name in sorted(names):
            info = header[name]
            begin, end = info["data_offsets"]
            entries[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + end - begin]}
            chunks.append((src, data_off + begin, end - begin))
            offset += end - begin
    blob = json.dumps(entries, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for src, start, length in chunks:
            with src.open("rb") as fin:
                fin.seek(start)
                remaining = length
                while remaining:
                    buf = fin.read(min(remaining, 1 << 24))
                    if not buf:
                        raise ValueError(f"{src.name}: truncated tensor data")
                    fout.write(buf)
                    remaining -= len(buf)
    return {k: dst.name for k in entries}


def rename_quant_config(q: dict) -> dict:
    """Prefix module names in a ModelOpt quantization config (quantized_layers keys, ignore / exclude_modules lists,
    config_groups targets). Values and every other key stay as they are."""
    out = dict(q)
    if "quantized_layers" in q:
        out["quantized_layers"] = {lm_name(k): v for k, v in q["quantized_layers"].items()}
    for key in ("ignore", "exclude_modules"):
        if q.get(key):
            out[key] = [lm_name(k) for k in q[key]]
    if isinstance(q.get("config_groups"), dict):
        groups = {}
        for g, body in q["config_groups"].items():
            body = dict(body)
            if body.get("targets"):
                body["targets"] = [lm_name(t) for t in body["targets"]]
            groups[g] = body
        out["config_groups"] = groups
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def reassemble(trunk: Path, upstream: Path, out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    tmp = out.with_name(out.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    t_idx = weight_map(trunk)
    u_idx = weight_map(upstream)
    t_cfg = json.loads(source_file(trunk, "config.json").read_text())
    u_cfg = json.loads(source_file(upstream, "config.json").read_text())
    if t_cfg.get("architectures") != ["NemotronHForCausalLM"]:
        raise SystemExit(f"trunk is not a standalone nemotron_h checkpoint: {t_cfg.get('architectures')}")
    if "vision_config" not in u_cfg or "sound_config" not in u_cfg or "llm_config" not in u_cfg:
        raise SystemExit("upstream config lacks vision_config / sound_config / llm_config")

    out_map: dict[str, str] = {}
    for shard in sorted(set(t_idx.values())):
        # weight_map() checked every shard name is one plain file name, so tmp / shard stays in tmp.
        out_map.update(write_renamed(source_file(trunk, shard), tmp / shard, lm_name))
    towers = sorted(k for k in u_idx if not k.startswith(LM))
    by_src: dict[Path, list[str]] = {}
    for k in towers:
        by_src.setdefault(source_file(upstream, u_idx[k]), []).append(k)
    tower_map = write_subset(by_src, tmp / SHARD_TOWERS)
    clash = set(tower_map) & set(out_map)
    if clash:
        raise SystemExit(f"tower and trunk names collide: {sorted(clash)[:3]}")
    out_map.update(tower_map)
    expected_lm = {lm_name(k) for k in t_idx}
    if {k for k in out_map if k.startswith(LM)} != expected_lm or len(tower_map) != len(towers):
        raise SystemExit("tensor count mismatch after reassembly")

    for p in sorted(upstream.rglob("*")):
        rel = p.relative_to(upstream)
        if (p.is_dir() or rel.parts[0].startswith(".") or rel.parts[0] == "media" or p.name in GENERATED
                or p.name in UPSTREAM_SKIP_NAMES or p.name.endswith(UPSTREAM_SKIP_SUFFIXES)):
            continue
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file(upstream, rel), tmp / rel)

    cfg = dict(u_cfg)
    cfg["quantization_config"] = rename_quant_config(t_cfg["quantization_config"])
    (tmp / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    hq = json.loads(source_file(trunk, "hf_quant_config.json").read_text())
    hq["quantization"] = rename_quant_config(hq["quantization"])
    (tmp / "hf_quant_config.json").write_text(json.dumps(hq, indent=4) + "\n")
    total = sum((tmp / s).stat().st_size for s in set(out_map.values()))
    (tmp / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(out_map.items()))}, indent=2) + "\n")
    files = sorted(p for p in tmp.rglob("*") if p.is_file())
    (tmp / "manifest.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(tmp)}\n" for p in files))
    tmp.rename(out)
    return {"lm_tensors": len(expected_lm), "tower_tensors": len(tower_map), "files": len(files) + 1,
            "quantized_layers": len(cfg["quantization_config"].get("quantized_layers", {})), "total_bytes": total}


def main() -> int:
    trunk, upstream, out = (Path(a) for a in sys.argv[1:4])
    try:
        r = reassemble(trunk, upstream, out)
    except CheckpointPathError as error:
        raise SystemExit(f"REASSEMBLE-FAILED: {error}") from None
    print(f"REASSEMBLE-OK {out} " + " ".join(f"{k}={v}" for k, v in r.items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
