"""Shard names from model.safetensors.index.json must stay inside the checkpoint.

make_split.py, split_mse.py, merge_lora.py, and apply_qwen38_abliteration.py
copy those names onto the source directory and the output directory. An
absolute name replaces the directory. A name with parent segments walks out of
it. The same name is opened and then handed to save_file, and the splitters
write it back into the new index.

These tests are the containment spec. They fail while that behavior is still
present, including when a check looks the path up and a later open looks it up
again.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[1]
MAKE_SPLIT = REPO / "models/nemotron-3.5-lightning-r1/eval/make_split.py"
SPLIT_MSE = REPO / "models/nemotron-3.5-lightning-r1/eval/split_mse.py"
MERGE_LORA = REPO / "models/jev-27b/scripts/merge_lora.py"
ABLITERAL = REPO / "scripts/qwen3_8/apply_qwen38_abliteration.py"
TOCTOU_LAUNCH = REPO / "tests/support/shard_toctou_launch.py"

SAFE_SHARD = "model-00001-of-00001.safetensors"
EARLY_SHARD = "!ok.safetensors"
LORA_KEY = "model.language_model.layers.0.mlp.down_proj.weight"
INSIDE_VALUE = 1.0
SECRET_VALUE = 42.0
SCRIPTS = ("make_split", "split_mse", "merge_lora")
SHA = "0" * 40

EXPERIMENT_YAML = f"""
schema_version: "1.0"
name: shard-path-test
source:
  model_id: Qwen/Qwen3.8-27B
  revision: "{SHA}"
abliteration:
  layer: 1
  seed: 1
  harmful_prompts: 1
  harmless_prompts: 1
  orthogonalize_harmless: false
  target_selectors: ["{LORA_KEY}"]
  expected_target_count: 1
datasets:
  harmful:
    source: local
    revision: "{SHA}"
quantization:
  scheme: NVFP4
  targets: Linear
  group_size: 16
  calibration_samples: 1
  max_sequence_length: 8
  calibration_dataset: local
  calibration_config: LLM
  pipeline: basic
  shard_size: 5GB
  ignore: ["lm_head", "mtp"]
  keep_bf16: ["mtp"]
validation:
  max_refusal_leakage: 0.01
  max_benign_kl_divergence: 0.05
  max_perplexity_delta_pct: 5.0
  vision_byte_identical: true
  mtp_present: true
runtime:
  kv_dtype: bf16
  context_length: 128
  compiled_mode: false
  flash_attention: false
  mtp_depth_initial: 1
  mtp_sweep_range: [1]
performance:
  target_tok_s: 1
  minimum_tok_s: 1
  warmup_repeats: 1
  measure_repeats: 1
  prompt_lengths: [8]
"""


def _weight() -> torch.Tensor:
    return torch.tensor([[1.0, 0.0], [0.0, 1.0]])


def _filled(value: float) -> torch.Tensor:
    return torch.full((2, 2), value)


def _write_shard(path: Path, tensors: dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path, metadata={"format": "pt"})


def _write_index(directory: Path, weight_map: dict[str, str]) -> None:
    payload = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    (directory / "model.safetensors.index.json").write_text(json.dumps(payload) + "\n")


def _fingerprint(path: Path) -> tuple[object, ...]:
    if not os.path.lexists(path):
        return ("missing",)
    info = path.lstat()
    if path.is_symlink():
        return ("symlink", os.readlink(path), info.st_ino)
    return ("file", info.st_ino, path.read_bytes())


def _safe_shard_name(name: str) -> bool:
    if name in ("", ".", ".."):
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    path = PurePosixPath(name)
    return path.name == name and not path.is_absolute()


def _contains_value(directory: Path, value: float) -> bool:
    if not directory.exists():
        return False
    for path in directory.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.suffix != ".safetensors":
            continue
        try:
            tensors = load_file(path)
        except Exception:
            continue
        for tensor in tensors.values():
            flat = tensor.detach().reshape(-1).float()
            if flat.numel() and bool(torch.any(flat == value)):
                return True
    return False


def _env(toctou_target: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    if toctou_target is not None:
        env["SHARD_TOCTOU_TARGET"] = str(toctou_target)
    return env


def _run(args: list[str], *, toctou_target: Path | None = None) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if toctou_target is not None:
        command.append(str(TOCTOU_LAUNCH))
    command.extend(args)
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=_env(toctou_target),
        check=False,
    )


def _output_names(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {path.name for path in directory.iterdir()}


def _assert_refused(
    proc: subprocess.CompletedProcess[str],
    out: Path,
    canary: Path,
    canary_fp: tuple[object, ...],
    *protected: tuple[Path, tuple[object, ...]],
) -> None:
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(canary) == canary_fp
    for path, fingerprint in protected:
        assert _fingerprint(path) == fingerprint, path
    if out.exists():
        assert EARLY_SHARD not in _output_names(out)
        assert not any(path.name == EARLY_SHARD for path in out.rglob("*"))
    index = out / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        assert all(_safe_shard_name(shard) for shard in weight_map.values()), weight_map
    assert _output_names(out) <= {"canary.txt"}


def _adapter(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(json.dumps({"r": 1, "lora_alpha": 1}) + "\n")
    (directory / "decision_head.json").write_text("{}\n")
    _write_shard(
        directory / "adapter_model.safetensors",
        {
            "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight": torch.zeros((1, 2)),
            "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight": torch.zeros((2, 1)),
        },
    )


def _g_file(directory: Path, streams: dict[str, float] | None = None) -> Path:
    path = directory / "variants.json"
    path.write_text(json.dumps({"mse": streams or {}}) + "\n")
    return path


def _invoke(
    kind: str,
    source: Path,
    out: Path,
    work: Path,
    *,
    toctou_target: Path | None = None,
    model: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if kind == "make_split":
        return _run([str(MAKE_SPLIT), str(source), str(out)], toctou_target=toctou_target)
    if kind == "split_mse":
        gpath = work / "variants.json"
        if not gpath.exists():
            _g_file(work)
        return _run(
            [str(SPLIT_MSE), str(source), str(gpath), str(out)],
            toctou_target=toctou_target,
        )
    if kind == "merge_lora":
        adapter = work / "adapter"
        _adapter(adapter)
        return _run(
            [
                str(MERGE_LORA),
                "--base",
                str(source),
                "--adapter",
                str(adapter),
                "--out",
                str(out),
            ],
            toctou_target=toctou_target,
        )
    if kind == "abliterate":
        config = work / "experiment.yaml"
        config.write_text(EXPERIMENT_YAML)
        direction = work / "direction"
        direction.mkdir(parents=True, exist_ok=True)
        _write_shard(direction / "direction.safetensors", {"direction": torch.zeros(2)})
        return _run(
            [
                str(ABLITERAL),
                "--config",
                str(config),
                "--run-root",
                str(out),
                "--direction-dir",
                str(direction),
                "--source-model",
                str(model or source),
            ],
            toctou_target=toctou_target,
        )
    raise AssertionError(kind)


def _prepare_out(work: Path) -> tuple[Path, Path, tuple[object, ...]]:
    out = work / "out"
    out.mkdir()
    canary = out / "canary.txt"
    canary.write_text("canary\n")
    return out, canary, _fingerprint(canary)


def _payload() -> dict[str, torch.Tensor]:
    return {LORA_KEY: _weight(), "other.input_scale": torch.tensor([2.0])}


@pytest.mark.parametrize("kind", SCRIPTS)
def test_absolute_shard_name_does_not_escape(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    victim = work / "victim.safetensors"
    _write_shard(victim, _payload())
    _write_shard(source / EARLY_SHARD, {"keep.weight": torch.tensor([3.0])})
    _write_index(
        source,
        {"keep.weight": EARLY_SHARD, LORA_KEY: str(victim), "other.input_scale": str(victim)},
    )
    out, canary, canary_fp = _prepare_out(work)
    victim_fp = _fingerprint(victim)
    proc = _invoke(kind, source, out, work)
    _assert_refused(proc, out, canary, canary_fp, (victim, victim_fp))


@pytest.mark.parametrize("kind", SCRIPTS)
def test_parent_segments_read_one_file_and_write_another(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    source = work / "a" / "b" / "source"
    source.mkdir(parents=True)
    out_parent = work / "a" / "out"
    out_parent.mkdir(parents=True)
    read_victim = work / "a" / "victim.safetensors"
    write_victim = work / "victim.safetensors"
    _write_shard(read_victim, _payload())
    write_victim.write_bytes(b"sentinel-outside-output")
    relative = "../../victim.safetensors"
    _write_shard(source / EARLY_SHARD, {"keep.weight": torch.tensor([3.0])})
    _write_index(
        source,
        {"keep.weight": EARLY_SHARD, LORA_KEY: relative, "other.input_scale": relative},
    )
    canary = out_parent / "canary.txt"
    canary.write_text("canary\n")
    read_fp = _fingerprint(read_victim)
    write_fp = _fingerprint(write_victim)
    canary_fp = _fingerprint(canary)
    proc = _invoke(kind, source, out_parent, work)
    _assert_refused(proc, out_parent, canary, canary_fp, (read_victim, read_fp), (write_victim, write_fp))


@pytest.mark.parametrize("kind", SCRIPTS)
def test_symlink_source_shard_resolving_outside_is_refused(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    secret = work / "secret.safetensors"
    _write_shard(secret, {LORA_KEY: _filled(SECRET_VALUE)})
    (source / SAFE_SHARD).symlink_to(secret)
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    out, canary, canary_fp = _prepare_out(work)
    secret_fp = _fingerprint(secret)
    proc = _invoke(kind, source, out, work)
    _assert_refused(proc, out, canary, canary_fp, (secret, secret_fp))
    assert not _contains_value(out, SECRET_VALUE)


@pytest.mark.parametrize("kind", SCRIPTS)
def test_source_symlink_is_not_reread_after_the_containment_check(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    inside = source / "inside.safetensors"
    secret = work / "secret.safetensors"
    _write_shard(inside, {LORA_KEY: _filled(INSIDE_VALUE)})
    _write_shard(secret, {LORA_KEY: _filled(SECRET_VALUE)})
    (source / SAFE_SHARD).symlink_to(inside)
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    out, _canary, _canary_fp = _prepare_out(work)
    secret_fp = _fingerprint(secret)
    proc = _invoke(kind, source, out, work, toctou_target=secret)
    assert _fingerprint(secret) == secret_fp
    assert not _contains_value(out, SECRET_VALUE), proc.stdout + proc.stderr
    if proc.returncode == 0:
        assert _contains_value(out, INSIDE_VALUE)
        index = out / "model.safetensors.index.json"
        if index.exists():
            shards = json.loads(index.read_text())["weight_map"].values()
            assert all(_safe_shard_name(shard) for shard in shards)


@pytest.mark.parametrize("kind", ("make_split", "merge_lora"))
def test_symlink_in_the_output_directory_resolving_outside_is_refused(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    _write_shard(source / SAFE_SHARD, _payload())
    _write_index(source, {LORA_KEY: SAFE_SHARD, "other.input_scale": SAFE_SHARD})
    out, canary, canary_fp = _prepare_out(work)
    victim = work / "victim.safetensors"
    victim.write_bytes(b"do-not-replace")
    link = out / SAFE_SHARD
    link.symlink_to(victim)
    link_fp = _fingerprint(link)
    victim_fp = _fingerprint(victim)
    proc = _invoke(kind, source, out, work)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(canary) == canary_fp
    assert _fingerprint(victim) == victim_fp
    assert _fingerprint(link) == link_fp


@pytest.mark.parametrize("kind", ("make_split", "merge_lora"))
def test_output_symlink_is_not_reopened_after_the_containment_check(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    _write_shard(source / SAFE_SHARD, {LORA_KEY: _filled(INSIDE_VALUE)})
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    out, _canary, _canary_fp = _prepare_out(work)
    placeholder = out / "placeholder.safetensors"
    placeholder.write_bytes(b"placeholder-inside-output")
    link = out / SAFE_SHARD
    link.symlink_to(placeholder)
    secret = work / "secret.safetensors"
    secret.write_bytes(b"secret-outside-output")
    placeholder_fp = _fingerprint(placeholder)
    secret_fp = _fingerprint(secret)
    proc = _invoke(kind, source, out, work, toctou_target=secret)
    assert _fingerprint(secret) == secret_fp
    if proc.returncode == 0:
        assert _fingerprint(placeholder) != placeholder_fp
    else:
        assert _fingerprint(placeholder) == placeholder_fp


@pytest.mark.parametrize("kind", SCRIPTS)
def test_nested_shard_name_is_refused(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    nested = "sub/model.safetensors"
    _write_shard(source / nested, _payload())
    _write_index(source, {LORA_KEY: nested, "other.input_scale": nested})
    out, canary, canary_fp = _prepare_out(work)
    (out / "sub").mkdir()
    proc = _invoke(kind, source, out, work)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(canary) == canary_fp
    assert not (out / nested).exists()
    index = out / "model.safetensors.index.json"
    if index.exists():
        shards = json.loads(index.read_text())["weight_map"].values()
        assert all(_safe_shard_name(shard) for shard in shards)


@pytest.mark.parametrize("kind", SCRIPTS)
def test_dot_slash_shard_name_is_refused(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    _write_shard(source / SAFE_SHARD, _payload())
    dotted = f"./{SAFE_SHARD}"
    _write_index(source, {LORA_KEY: dotted, "other.input_scale": dotted})
    out, canary, canary_fp = _prepare_out(work)
    proc = _invoke(kind, source, out, work)
    _assert_refused(proc, out, canary, canary_fp)


@pytest.mark.parametrize("kind", SCRIPTS)
def test_backslash_shard_name_is_refused(tmp_path: Path, kind: str) -> None:
    work = tmp_path / kind
    work.mkdir()
    source = work / "source"
    source.mkdir()
    name = "model\\shard.safetensors"
    _write_shard(source / name, _payload())
    _write_index(source, {LORA_KEY: name, "other.input_scale": name})
    out, canary, canary_fp = _prepare_out(work)
    proc = _invoke(kind, source, out, work)
    _assert_refused(proc, out, canary, canary_fp)


def test_make_split_keeps_a_single_component_shard_inside_the_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    weight = torch.tensor([1.0])
    scale = torch.tensor([2.0])
    _write_shard(source / SAFE_SHARD, {"keep.weight": weight, "keep.input_scale": scale})
    _write_index(source, {"keep.weight": SAFE_SHARD, "keep.input_scale": SAFE_SHARD})
    (source / "tokenizer_config.json").write_text("{}\n")
    untouched = tmp_path / "untouched.txt"
    untouched.write_text("untouched\n")
    untouched_fp = _fingerprint(untouched)
    proc = _invoke("make_split", source, out, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _fingerprint(untouched) == untouched_fp
    written = load_file(out / SAFE_SHARD)
    scales = load_file(out / "model-input-scales.safetensors")
    assert torch.equal(written["keep.weight"], weight)
    assert torch.equal(scales["keep.input_scale"], scale)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {
        "keep.input_scale": "model-input-scales.safetensors",
        "keep.weight": SAFE_SHARD,
    }
    assert (out / "tokenizer_config.json").read_text() == "{}\n"


def test_split_mse_keeps_a_single_component_shard_inside_the_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    tensors: dict[str, torch.Tensor] = {
        f"backbone.layers.0.mixer.experts.{i}.up_proj.input_scale": torch.tensor([2.0])
        for i in range(5934)
    }
    regular = torch.tensor([4.0])
    tensors["backbone.norm.weight"] = regular
    _write_shard(source / SAFE_SHARD, tensors)
    _write_index(source, {name: SAFE_SHARD for name in tensors})
    (source / "tokenizer_config.json").write_text("{}\n")
    _g_file(tmp_path, {"L0.up": 2688.0})
    proc = _invoke("split_mse", source, out, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    written = load_file(out / SAFE_SHARD)
    scales = load_file(out / "model-input-scales.safetensors")
    assert list(written) == ["backbone.norm.weight"]
    assert torch.equal(written["backbone.norm.weight"], regular)
    assert len(scales) == 5934
    assert torch.equal(next(iter(scales.values())), torch.tensor([1.0]))
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"].values()) <= {SAFE_SHARD, "model-input-scales.safetensors"}
    assert all(_safe_shard_name(shard) for shard in index["weight_map"].values())
    assert (out / "tokenizer_config.json").read_text() == "{}\n"


def test_merge_lora_keeps_a_single_component_shard_inside_the_output(tmp_path: Path) -> None:
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    weight = _weight()
    _write_shard(source / SAFE_SHARD, {LORA_KEY: weight})
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    (source / "config.json").write_text('{"hidden": 2}\n')
    proc = _invoke("merge_lora", source, out, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    merged = load_file(out / SAFE_SHARD)
    assert torch.equal(merged[LORA_KEY], weight)
    index = json.loads((out / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {LORA_KEY: SAFE_SHARD}
    assert (out / "decision_head.json").read_text() == "{}\n"
    assert (out / "config.json").read_text() == '{"hidden": 2}\n'


def test_abliteration_absolute_shard_name_does_not_escape(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    victim = tmp_path / "victim.safetensors"
    _write_shard(victim, {LORA_KEY: _weight()})
    _write_shard(source / EARLY_SHARD, {"keep.weight": torch.tensor([3.0])})
    _write_index(source, {"keep.weight": EARLY_SHARD, LORA_KEY: str(victim)})
    run_root = tmp_path / "run"
    run_root.mkdir()
    victim_fp = _fingerprint(victim)
    proc = _invoke("abliterate", source, run_root, tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(victim) == victim_fp
    assert not any(path.name == EARLY_SHARD for path in run_root.rglob(EARLY_SHARD))
    for index in run_root.rglob("model.safetensors.index.json"):
        shards = json.loads(index.read_text())["weight_map"].values()
        assert all(_safe_shard_name(shard) for shard in shards)


def test_abliteration_parent_segments_do_not_write_the_escaped_file(tmp_path: Path) -> None:
    source = tmp_path / "run" / "m" / "source"
    source.mkdir(parents=True)
    read_victim = tmp_path / "run" / "victim.safetensors"
    write_victim = tmp_path / "victim.safetensors"
    _write_shard(read_victim, {LORA_KEY: _weight()})
    write_victim.write_bytes(b"sentinel-outside-run")
    relative = "../../victim.safetensors"
    _write_shard(source / EARLY_SHARD, {"keep.weight": torch.tensor([3.0])})
    _write_index(source, {"keep.weight": EARLY_SHARD, LORA_KEY: relative})
    run_root = tmp_path / "run"
    read_fp = _fingerprint(read_victim)
    write_fp = _fingerprint(write_victim)
    proc = _invoke("abliterate", source, run_root, tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(read_victim) == read_fp
    assert _fingerprint(write_victim) == write_fp


def test_abliteration_symlink_source_resolving_outside_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    secret = tmp_path / "secret.safetensors"
    _write_shard(secret, {LORA_KEY: _filled(SECRET_VALUE)})
    (source / SAFE_SHARD).symlink_to(secret)
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    run_root = tmp_path / "run"
    run_root.mkdir()
    secret_fp = _fingerprint(secret)
    proc = _invoke("abliterate", source, run_root, tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert _fingerprint(secret) == secret_fp
    assert not _contains_value(run_root, SECRET_VALUE)


def test_abliteration_source_symlink_is_not_reread_after_the_check(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    inside = source / "inside.safetensors"
    secret = tmp_path / "secret.safetensors"
    _write_shard(inside, {LORA_KEY: _filled(INSIDE_VALUE)})
    _write_shard(secret, {LORA_KEY: _filled(SECRET_VALUE)})
    (source / SAFE_SHARD).symlink_to(inside)
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    run_root = tmp_path / "run"
    run_root.mkdir()
    secret_fp = _fingerprint(secret)
    proc = _invoke("abliterate", source, run_root, tmp_path, toctou_target=secret)
    assert _fingerprint(secret) == secret_fp
    assert not _contains_value(run_root, SECRET_VALUE), proc.stdout + proc.stderr
    if proc.returncode == 0:
        assert _contains_value(run_root, INSIDE_VALUE)


@pytest.mark.parametrize("name", [f"./{SAFE_SHARD}", "model\\shard.safetensors"])
def test_abliteration_rejects_separator_in_shard_name(tmp_path: Path, name: str) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shard_path = source / SAFE_SHARD if name.startswith("./") else source / name
    _write_shard(shard_path, {LORA_KEY: _weight()})
    _write_index(source, {LORA_KEY: name})
    run_root = tmp_path / "run"
    run_root.mkdir()
    proc = _invoke("abliterate", source, run_root, tmp_path)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    for index in run_root.rglob("model.safetensors.index.json"):
        shards = json.loads(index.read_text())["weight_map"].values()
        assert all(_safe_shard_name(shard) for shard in shards)


def test_abliteration_keeps_a_single_component_shard_inside_the_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    weight = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    _write_shard(source / SAFE_SHARD, {LORA_KEY: weight})
    _write_index(source, {LORA_KEY: SAFE_SHARD})
    (source / "config.json").write_text("{}\n")
    run_root = tmp_path / "run"
    direction = tmp_path / "direction"
    direction.mkdir()
    _write_shard(direction / "direction.safetensors", {"direction": torch.tensor([1.0, 0.0])})
    config = tmp_path / "experiment.yaml"
    config.write_text(EXPERIMENT_YAML)
    proc = _run(
        [
            str(ABLITERAL),
            "--config",
            str(config),
            "--run-root",
            str(run_root),
            "--direction-dir",
            str(direction),
            "--source-model",
            str(source),
        ]
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    stage = run_root / "apply_abliteration"
    written = load_file(stage / SAFE_SHARD)
    expected = torch.tensor([[0.0, 0.0], [5.0, 6.0]])
    assert torch.allclose(written[LORA_KEY].float(), expected)
    index = json.loads((stage / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {LORA_KEY: SAFE_SHARD}
    assert (stage / "config.json").read_text() == "{}\n"
