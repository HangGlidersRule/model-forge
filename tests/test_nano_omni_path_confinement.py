"""Issue #35 / PR #99: the Nano-Omni repair scripts read only what lies inside their input checkpoint directories.

An index shard name must be one plain file name, and every file a script reads (shards, configs, sidecars it copies)
must resolve to a regular file inside its checkpoint directory, or inside the Hub cache's blobs/ when the directory is a
Hub snapshot (whose files are all links into ../../blobs). Reproduces the review's probes on #99: an absolute shard
name that truncated a file outside both snapshots, a ../ name that read and wrote outside the snapshot and output
directories, and a sidecar link to a file outside the snapshot copied in as a regular file. Synthetic, stdlib only."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "models" / "nemotron-3-nano-omni-r1" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _st(path: Path, tensors: dict[str, tuple[str, list, bytes]]) -> None:
    header, off, data = {}, 0, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        data += raw
    blob = json.dumps(header).encode()
    blob += b" " * (-len(blob) % 8)
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def _trunk(root: Path, shard: str = "model-00001-of-00001.safetensors") -> None:
    root.mkdir(parents=True)
    q = {"backbone.layers.1.mixer.experts.0.up_proj": {"quant_algo": "W4A16_NVFP4"}, "lm_head": {"quant_algo": "W4A16_NVFP4"}}
    _st(root / "model-00001-of-00001.safetensors", {"backbone.embeddings.weight": ("U8", [3], b"emb"),
                                                    "lm_head.weight": ("U8", [4], b"head")})
    wm = {"backbone.embeddings.weight": shard, "lm_head.weight": shard}
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (root / "config.json").write_text(json.dumps({"architectures": ["NemotronHForCausalLM"], "quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": q, "ignore": []}}))
    (root / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "MIXED_PRECISION",
                                                                            "quantized_layers": q}}))


def _upstream(root: Path, tower_shard: str = "model-00001-of-00001.safetensors") -> None:
    root.mkdir(parents=True)
    _st(root / "model-00001-of-00001.safetensors", {"vision_model.w": ("U8", [5], b"radio"),
                                                    "language_model.lm_head.weight": ("U8", [4], b"bf16")})
    wm = {"vision_model.w": tower_shard, "language_model.lm_head.weight": "model-00001-of-00001.safetensors"}
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    (root / "config.json").write_text(json.dumps({"architectures": ["NemotronH_Nano_Omni_Reasoning_V3"],
                                                  "vision_config": {}, "sound_config": {}, "llm_config": {}}))
    (root / "tokenizer.json").write_text("{}")


def _victim(tmp_path: Path) -> Path:
    victim = tmp_path / "elsewhere" / "victim.safetensors"
    victim.parent.mkdir()
    _st(victim, {"secret": ("U8", [6], b"secret")})
    return victim


# --- the shard-name rule ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", ".", "..", "../x.safetensors", "a/b.safetensors", "/abs.safetensors",
                                 "a\\b.safetensors", "x\x00.safetensors", 7, None])
def test_shard_name_accepts_only_one_plain_file_name(bad: object) -> None:
    cp = _load("checkpoint_paths")
    with pytest.raises(cp.CheckpointPathError):
        cp.shard_name(bad)
    assert cp.shard_name("model-00001-of-00002.safetensors") == "model-00001-of-00002.safetensors"


# --- reassemble_omni_wrapper.py --------------------------------------------------------------------------------------

def test_reassembly_refuses_an_absolute_trunk_shard_and_leaves_the_target_intact(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    victim = _victim(tmp_path)
    before = victim.read_bytes()
    _trunk(tmp_path / "trunk", shard=str(victim))
    _upstream(tmp_path / "up")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert victim.read_bytes() == before
    assert not (tmp_path / "out").exists()


def test_reassembly_refuses_a_dotdot_trunk_shard(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _victim(tmp_path)
    _trunk(tmp_path / "trunk", shard="../elsewhere/victim.safetensors")
    _upstream(tmp_path / "up")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "elsewhere" / "victim.safetensors.partial").exists()
    assert sorted(p.name for p in (tmp_path / "elsewhere").iterdir()) == ["victim.safetensors"]


def test_reassembly_refuses_a_dotdot_tower_shard(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _victim(tmp_path)
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up", tower_shard="../elsewhere/victim.safetensors")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")


def test_reassembly_refuses_an_upstream_file_linked_outside_the_snapshot(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    secret = tmp_path / "elsewhere" / "outside.txt"
    secret.parent.mkdir()
    secret.write_text("not for publication")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    os.symlink(secret, tmp_path / "up" / "processor_config.json")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reassembly_refuses_a_trunk_config_linked_outside(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    outside = tmp_path / "elsewhere.json"
    (tmp_path / "trunk" / "hf_quant_config.json").rename(outside)
    os.symlink(outside, tmp_path / "trunk" / "hf_quant_config.json")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")


def test_reassembly_reads_a_hub_cache_snapshot(tmp_path: Path) -> None:
    """A Hub cache snapshot is all links into ../../blobs: that layout must keep working."""
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    flat = tmp_path / "flat"
    _upstream(flat)
    repo = tmp_path / "hub" / "models--nvidia--omni"
    snap, blobs = repo / "snapshots" / ("a" * 40), repo / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir()
    for f in flat.iterdir():
        blob = blobs / f"blob-{f.name}"
        f.rename(blob)
        os.symlink(os.path.relpath(blob, snap), snap / f.name)
    r = rea.reassemble(tmp_path / "trunk", snap, tmp_path / "out")
    assert r["tower_tensors"] == 1
    assert (tmp_path / "out" / "tokenizer.json").read_text() == "{}"
    assert not (tmp_path / "out" / "tokenizer.json").is_symlink()


def test_reassembly_refuses_a_nested_shard_name_inside_the_trunk(tmp_path: Path) -> None:
    """Even a file inside the trunk: an index shard name must be one plain file name."""
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk", shard="sub/model-00001-of-00001.safetensors")
    (tmp_path / "trunk" / "sub").mkdir()
    (tmp_path / "trunk" / "model-00001-of-00001.safetensors").rename(
        tmp_path / "trunk" / "sub" / "model-00001-of-00001.safetensors")
    _upstream(tmp_path / "up")
    with pytest.raises(rea.CheckpointPathError, match="single file name"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")


def test_reassembly_refuses_a_shard_that_is_a_directory(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk", shard="shards.d")
    (tmp_path / "trunk" / "shards.d").mkdir()
    _upstream(tmp_path / "up")
    with pytest.raises(rea.CheckpointPathError, match="not a regular file"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")


def test_only_a_hub_snapshot_may_link_into_a_blobs_directory(tmp_path: Path) -> None:
    """A plain directory two levels under a blobs/ sibling gets no blobs allowance; only <repo>/snapshots/<rev> does."""
    cp = _load("checkpoint_paths")
    base = tmp_path / "repo"
    plain, blobs = base / "work" / "ckpt", base / "blobs"
    plain.mkdir(parents=True)
    blobs.mkdir()
    (blobs / "b1").write_text("bytes")
    os.symlink(os.path.relpath(blobs / "b1", plain), plain / "config.json")
    with pytest.raises(cp.CheckpointPathError, match="outside"):
        cp.source_file(plain, "config.json")
    snap = base / "snapshots" / "rev"
    snap.mkdir(parents=True)
    os.symlink(os.path.relpath(blobs / "b1", snap), snap / "config.json")
    assert cp.source_file(snap, "config.json").read_text() == "bytes"


def test_reassembly_refuses_a_trunk_shard_linked_outside(tmp_path: Path) -> None:
    """A plainly named trunk shard that links outside the trunk is refused, not read."""
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    shard = tmp_path / "trunk" / "model-00001-of-00001.safetensors"
    outside = tmp_path / "outside-trunk.safetensors"
    shard.rename(outside)
    os.symlink(outside, shard)
    with pytest.raises(rea.CheckpointPathError, match="outside"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_reassembly_refuses_an_upstream_shard_linked_outside(tmp_path: Path) -> None:
    """A plain shard name passes the name rule; the file it names must still resolve inside the snapshot. Without the
    check the outside file's tensor of the same name would be published as the vision tower."""
    rea = _load("reassemble_omni_wrapper")
    stolen = tmp_path / "elsewhere" / "other.safetensors"
    stolen.parent.mkdir()
    _st(stolen, {"vision_model.w": ("U8", [5], b"STOLE")})
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    shard = tmp_path / "up" / "model-00001-of-00001.safetensors"
    shard.unlink()
    os.symlink(stolen, shard)
    with pytest.raises(rea.CheckpointPathError, match="outside"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reassembly_refuses_a_nested_upstream_shard_name(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up", tower_shard="sub/model-00001-of-00001.safetensors")
    (tmp_path / "up" / "sub").mkdir()
    (tmp_path / "up" / "sub" / "model-00001-of-00001.safetensors").write_bytes(
        (tmp_path / "up" / "model-00001-of-00001.safetensors").read_bytes())
    with pytest.raises(rea.CheckpointPathError, match="single file name"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")


def test_reassembly_cli_reports_a_containment_failure(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _victim(tmp_path)
    _trunk(tmp_path / "trunk", shard="../elsewhere/victim.safetensors")
    _upstream(tmp_path / "up")
    argv = sys.argv
    try:
        sys.argv = ["r", str(tmp_path / "trunk"), str(tmp_path / "up"), str(tmp_path / "out")]
        with pytest.raises(SystemExit, match="REASSEMBLE-FAILED"):
            rea.main()
    finally:
        sys.argv = argv


# --- restore_router_bias.py ------------------------------------------------------------------------------------------

def _bias_ckpt(root: Path, shard: str = "model-1.safetensors") -> None:
    root.mkdir(parents=True)
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    _st(root / "model-1.safetensors", {name: ("F32", [1], struct.pack("<f", 55.75))})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {name: shard}}))
    (root / "config.json").write_text("{}")


def _bias_up(root: Path) -> None:
    root.mkdir(parents=True)
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    _st(root / "u.safetensors", {name: ("F32", [1], struct.pack("<f", 55.7645))})
    (root / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {name: "u.safetensors"}}))


def _run(mod, *args: Path) -> int:
    argv = sys.argv
    try:
        sys.argv = ["r", *map(str, args)]
        return mod.main()
    finally:
        sys.argv = argv


@pytest.mark.parametrize("shard", ["../elsewhere/victim.safetensors", "ABSOLUTE"])
def test_restore_refuses_a_shard_name_outside_the_checkpoint(tmp_path: Path, shard: str) -> None:
    mod = _load("restore_router_bias")
    victim = _victim(tmp_path)
    _bias_up(tmp_path / "up")
    _bias_ckpt(tmp_path / "ck", shard=str(victim) if shard == "ABSOLUTE" else shard)
    with pytest.raises(SystemExit, match="RESTORE-FAILED"):
        _run(mod, tmp_path / "up", tmp_path / "ck", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_restore_refuses_a_checkpoint_file_linked_outside(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    secret = tmp_path / "elsewhere" / "outside-config.json"
    secret.parent.mkdir()
    secret.write_text("{}")
    _bias_up(tmp_path / "up")
    _bias_ckpt(tmp_path / "ck")
    os.symlink(secret, tmp_path / "ck" / "tokenizer_config.json")
    with pytest.raises(SystemExit, match="RESTORE-FAILED"):
        _run(mod, tmp_path / "up", tmp_path / "ck", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_restore_read_refuses_a_shard_linked_outside(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    outside = tmp_path / "elsewhere" / "other.safetensors"
    outside.parent.mkdir()
    _st(outside, {name: ("F32", [1], struct.pack("<f", 1.0))})
    _bias_ckpt(tmp_path / "ck")
    (tmp_path / "ck" / "model-1.safetensors").unlink()
    os.symlink(outside, tmp_path / "ck" / "model-1.safetensors")
    wm = mod.checkpoint_map(tmp_path / "ck")
    with pytest.raises(mod.CheckpointPathError, match="outside"):
        mod.read(tmp_path / "ck", wm, name)


def test_restore_refuses_a_nested_shard_name(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    _bias_up(tmp_path / "up")
    _bias_ckpt(tmp_path / "ck", shard="sub/model-1.safetensors")
    (tmp_path / "ck" / "sub").mkdir()
    (tmp_path / "ck" / "model-1.safetensors").rename(tmp_path / "ck" / "sub" / "model-1.safetensors")
    with pytest.raises(SystemExit, match="single file name"):
        _run(mod, tmp_path / "up", tmp_path / "ck", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_restore_still_restores_inside_the_checkpoint(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    _bias_up(tmp_path / "up")
    _bias_ckpt(tmp_path / "ck")
    assert _run(mod, tmp_path / "up", tmp_path / "ck", tmp_path / "out") == 0
    wm = mod.checkpoint_map(tmp_path / "out")
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    assert mod.read(tmp_path / "out", wm, name)[2] == struct.pack("<f", 55.7645)


# --- normalize_quant_names.py ----------------------------------------------------------------------------------------

def test_normalize_refuses_a_trunk_file_linked_outside(tmp_path: Path) -> None:
    mod = _load("normalize_quant_names")
    secret = tmp_path / "elsewhere" / "outside-tokenizer.json"
    secret.parent.mkdir()
    secret.write_text("x")
    _trunk(tmp_path / "trunk")
    _trunk(tmp_path / "ref")
    os.symlink(secret, tmp_path / "trunk" / "tokenizer.json")
    with pytest.raises(SystemExit, match="NORMALIZE-FAILED"):
        _run(mod, tmp_path / "trunk", tmp_path / "ref", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


@pytest.mark.parametrize("script", ["reassemble", "restore", "normalize"])
def test_a_failed_write_leaves_no_partial_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, script: str) -> None:
    """A run that fails while writing (here: every file copy raises) removes its OUT.partial."""
    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    out = tmp_path / "out"
    if script == "reassemble":
        mod = _load("reassemble_omni_wrapper")
        _trunk(tmp_path / "trunk")
        _upstream(tmp_path / "up")
        monkeypatch.setattr(mod.shutil, "copyfile", boom)
        with pytest.raises(OSError, match="simulated"):
            mod.reassemble(tmp_path / "trunk", tmp_path / "up", out)
    elif script == "restore":
        mod = _load("restore_router_bias")
        _bias_up(tmp_path / "up")
        _bias_ckpt(tmp_path / "ck")
        monkeypatch.setattr(mod.shutil, "copyfile", boom)
        with pytest.raises(OSError, match="simulated"):
            _run(mod, tmp_path / "up", tmp_path / "ck", out)
    else:
        mod = _load("normalize_quant_names")
        _trunk(tmp_path / "trunk")
        _trunk(tmp_path / "ref")
        monkeypatch.setattr(mod.shutil, "copyfile", boom)
        with pytest.raises(OSError, match="simulated"):
            _run(mod, tmp_path / "trunk", tmp_path / "ref", out)
    assert not out.exists() and not (tmp_path / "out.partial").exists()


def test_normalize_still_copies_a_plain_trunk(tmp_path: Path) -> None:
    mod = _load("normalize_quant_names")
    _trunk(tmp_path / "trunk")
    _trunk(tmp_path / "ref")
    assert _run(mod, tmp_path / "trunk", tmp_path / "ref", tmp_path / "out") == 0
    assert (tmp_path / "out" / "model-00001-of-00001.safetensors").read_bytes() == \
        (tmp_path / "trunk" / "model-00001-of-00001.safetensors").read_bytes()


# --- round 2: every source_file() call site and the exact allowance ------------------------------------------------

def _link_out(root: Path, name: str, tmp_path: Path, content: str) -> None:
    """Replace root/name with a link to a file outside root holding `content`."""
    outside = tmp_path / "outside" / f"{root.name}-{name}"
    outside.parent.mkdir(exist_ok=True)
    outside.write_text(content)
    (root / name).unlink()
    os.symlink(outside, root / name)


@pytest.mark.parametrize("side", ["trunk", "up"])
def test_reassembly_refuses_a_config_linked_outside(tmp_path: Path, side: str) -> None:
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    _link_out(tmp_path / side, "config.json", tmp_path, (tmp_path / side / "config.json").read_text())
    with pytest.raises(rea.CheckpointPathError, match="outside"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


@pytest.mark.parametrize("name", ["hf_quant_config.json", "config.json"])
@pytest.mark.parametrize("side", ["trunk", "ref"])
def test_normalize_refuses_a_parsed_file_linked_outside(tmp_path: Path, name: str, side: str) -> None:
    mod = _load("normalize_quant_names")
    _trunk(tmp_path / "trunk")
    _trunk(tmp_path / "ref")
    _link_out(tmp_path / side, name, tmp_path, (tmp_path / side / name).read_text())
    with pytest.raises(SystemExit, match="NORMALIZE-FAILED.*outside"):
        _run(mod, tmp_path / "trunk", tmp_path / "ref", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_containment_is_by_path_not_by_name_prefix(tmp_path: Path) -> None:
    """A sibling whose name extends the checkpoint's (ck-other next to ck) is outside it."""
    cp = _load("checkpoint_paths")
    ck, sibling = tmp_path / "ck", tmp_path / "ck-other"
    ck.mkdir()
    sibling.mkdir()
    (sibling / "config.json").write_text("{}")
    os.symlink(sibling / "config.json", ck / "config.json")
    with pytest.raises(cp.CheckpointPathError, match="outside"):
        cp.source_file(ck, "config.json")


def _hub_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "hub" / "models--org--name"
    snap = repo / "snapshots" / ("a" * 40)
    snap.mkdir(parents=True)
    (repo / "blobs").mkdir()
    return repo, snap


def test_snapshot_links_may_reach_only_the_repositorys_blobs(tmp_path: Path) -> None:
    cp = _load("checkpoint_paths")
    repo, snap = _hub_snapshot(tmp_path)
    (repo / "blobs" / "ok").write_text("blob")
    os.symlink(os.path.relpath(repo / "blobs" / "ok", snap), snap / "good.json")
    assert cp.source_file(snap, "good.json").read_text() == "blob"
    # A file in the repository directory but outside blobs/ (refs/main here).
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    os.symlink(os.path.relpath(repo / "refs" / "main", snap), snap / "refs.json")
    with pytest.raises(cp.CheckpointPathError, match="outside"):
        cp.source_file(snap, "refs.json")
    # Another repository's blob in the same cache.
    other = tmp_path / "hub" / "models--org--other" / "blobs"
    other.mkdir(parents=True)
    (other / "theirs").write_text("not ours")
    os.symlink(os.path.relpath(other / "theirs", snap), snap / "theirs.json")
    with pytest.raises(cp.CheckpointPathError, match="outside"):
        cp.source_file(snap, "theirs.json")


def test_weight_map_reads_the_index_inside_the_checkpoint(tmp_path: Path) -> None:
    cp = _load("checkpoint_paths")
    _trunk(tmp_path / "trunk")
    _link_out(tmp_path / "trunk", "model.safetensors.index.json", tmp_path,
              (tmp_path / "trunk" / "model.safetensors.index.json").read_text())
    with pytest.raises(cp.CheckpointPathError, match="outside"):
        cp.weight_map(tmp_path / "trunk")


@pytest.mark.parametrize("body", ['{"weight_map": ["a"]}', '["weight_map"]', '{"metadata": {}}'])
def test_weight_map_refuses_an_index_without_a_mapping(tmp_path: Path, body: str) -> None:
    cp = _load("checkpoint_paths")
    (tmp_path / "ck").mkdir()
    (tmp_path / "ck" / "model.safetensors.index.json").write_text(body)
    with pytest.raises(cp.CheckpointPathError, match="no weight_map mapping"):
        cp.weight_map(tmp_path / "ck")


def test_restore_refuses_an_upstream_shard_linked_outside(tmp_path: Path) -> None:
    """The upstream float32 listing reads every upstream shard's header, and reads it from inside the upstream
    directory: a second shard holding no float32 tensor (so no later read touches it) is still refused."""
    mod = _load("restore_router_bias")
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    up = tmp_path / "up"
    up.mkdir()
    _st(up / "u.safetensors", {name: ("F32", [1], struct.pack("<f", 55.7645))})
    outside = tmp_path / "outside-v.safetensors"
    _st(outside, {"language_model.backbone.layers.2.mixer.gate.weight": ("BF16", [1], b"\x00\x01")})
    os.symlink(outside, up / "v.safetensors")
    wm = {name: "u.safetensors", "language_model.backbone.layers.2.mixer.gate.weight": "v.safetensors"}
    (up / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    _bias_ckpt(tmp_path / "ck")
    with pytest.raises(SystemExit, match="RESTORE-FAILED.*outside"):
        _run(mod, up, tmp_path / "ck", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_restore_refuses_a_single_file_checkpoint_linked_outside(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    name = "language_model.backbone.layers.1.mixer.gate.e_score_correction_bias"
    ck = tmp_path / "ck"
    ck.mkdir()
    outside = tmp_path / "outside-model.safetensors"
    _st(outside, {name: ("F32", [1], struct.pack("<f", 55.75))})
    os.symlink(outside, ck / "model.safetensors")
    with pytest.raises(mod.CheckpointPathError, match="outside"):
        mod.checkpoint_map(ck)


# --- round 2: no output file is written twice; a refused input leaves nothing behind -------------------------------

@pytest.mark.parametrize("shard", ["model-towers.safetensors", "tokenizer.json", "config.json"])
def test_reassembly_refuses_a_trunk_shard_named_like_another_output(tmp_path: Path, shard: str) -> None:
    """The towers shard, a file copied from upstream and a generated file would each overwrite such a shard."""
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk", shard=shard)
    if shard != "config.json":  # the trunk's own config.json already exists under that name
        (tmp_path / "trunk" / "model-00001-of-00001.safetensors").rename(tmp_path / "trunk" / shard)
    _upstream(tmp_path / "up")
    with pytest.raises(SystemExit, match="collide with output files"):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.partial").exists()


def test_reassembly_refuses_a_linked_sidecar_before_writing_anything(tmp_path: Path) -> None:
    rea = _load("reassemble_omni_wrapper")
    _trunk(tmp_path / "trunk")
    _upstream(tmp_path / "up")
    _link_out(tmp_path / "up", "tokenizer.json", tmp_path, "{}")
    with pytest.raises(rea.CheckpointPathError):
        rea.reassemble(tmp_path / "trunk", tmp_path / "up", tmp_path / "out")
    assert not (tmp_path / "out.partial").exists()


def test_restore_refuses_a_linked_file_before_writing_anything(tmp_path: Path) -> None:
    mod = _load("restore_router_bias")
    _bias_up(tmp_path / "up")
    _bias_ckpt(tmp_path / "ck")
    _link_out(tmp_path / "ck", "config.json", tmp_path, "{}")
    with pytest.raises(SystemExit, match="RESTORE-FAILED"):
        _run(mod, tmp_path / "up", tmp_path / "ck", tmp_path / "out")
    assert not (tmp_path / "out.partial").exists()


# --- build_omni_calib.py: CALIB-OK only for the pinned set --------------------------------------------------------

def _calib():
    spec = importlib.util.spec_from_file_location(
        "build_omni_calib", SCRIPTS.parent / "calibration" / "build_omni_calib.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibration_pin_accepts_only_the_pinned_set() -> None:
    mod = _calib()
    assert mod.pin_mismatches(576, 320, mod.EXPECTED_SHA256) == []
    assert mod.pin_mismatches(575, 320, mod.EXPECTED_SHA256) == ["rows=575, pinned 576"]
    assert mod.pin_mismatches(576, 319, mod.EXPECTED_SHA256) == ["media=319, pinned 320"]
    assert len(mod.pin_mismatches(576, 320, "0" * 64)) == 1


def test_calibration_finish_refuses_a_short_set(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The message names each measured value: one row, no media, and this output's own digest."""
    mod = _calib()
    (tmp_path / "media").mkdir()
    rows = [{"stream": "text", "messages": [{"role": "user", "content": "q"}], "source": "s", "row": 0}]
    with pytest.raises(SystemExit, match="CALIB-FAILED") as failure:
        mod.finish(tmp_path, rows)
    message = str(failure.value)
    digest = hashlib.sha256((tmp_path / "calib.jsonl").read_bytes()).hexdigest()
    assert "rows=1," in message and "media=0," in message and f"sha256={digest}," in message
    assert "CALIB-OK" not in capsys.readouterr().out


def test_calibration_finish_measures_each_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A set wrong in exactly one way names exactly that mismatch: the row count, the media count and the digest
    are each measured from the output, never taken from the pin."""
    mod = _calib()
    media = tmp_path / "media"
    media.mkdir()
    for i in range(mod.EXPECTED_MEDIA):
        (media / f"m{i}").write_bytes(b"x")
    rows = [{"stream": "text", "messages": [], "source": "s", "row": i} for i in range(mod.EXPECTED_ROWS)]
    with pytest.raises(SystemExit) as failure:
        mod.finish(tmp_path, rows)
    message = str(failure.value)
    assert "rows=" not in message and "media=" not in message and "sha256=" in message
    (media / "m0").unlink()
    with pytest.raises(SystemExit) as failure:
        mod.finish(tmp_path, rows)
    assert f"media={mod.EXPECTED_MEDIA - 1}," in str(failure.value) and "rows=" not in str(failure.value)
    with pytest.raises(SystemExit) as failure:
        mod.finish(tmp_path, rows[:-1])
    assert f"rows={mod.EXPECTED_ROWS - 1}," in str(failure.value)
    assert "CALIB-OK" not in capsys.readouterr().out
